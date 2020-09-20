# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Module for constructing a linear-chain CRF.
The following snippet is an example of a CRF layer on top of a batched sequence
of unary scores (logits for every word). This example also decodes the most
likely sequence at test time. There are two ways to do decoding. One
is using crf_decode to do decoding in Tensorflow , and the other one is using
viterbi_decode in Numpy.
log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
    unary_scores, gold_tags, sequence_lengths)
loss = tf.reduce_mean(-log_likelihood)
train_op = tf.compat.v1.train.GradientDescentOptimizer(0.01).minimize(loss)
# Decoding in Tensorflow.
viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(
    unary_scores, transition_params, sequence_lengths)
tf_viterbi_sequence, tf_viterbi_score, _ = session.run(
    [viterbi_sequence, viterbi_score, train_op])
# Decoding in Numpy.
tf_unary_scores, tf_sequence_lengths, tf_transition_params, _ = session.run(
    [unary_scores, sequence_lengths, transition_params, train_op])
for tf_unary_scores_, tf_sequence_length_ in zip(tf_unary_scores,
                                                 tf_sequence_lengths):
    # Remove padding.
    tf_unary_scores_ = tf_unary_scores_[:tf_sequence_length_]
    # Compute the highest score and its tag sequence.
    tf_viterbi_sequence, tf_viterbi_score = tf.contrib.crf.viterbi_decode(
        tf_unary_scores_, tf_transition_params)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variable_scope as vs

__all__ = [
    "crf_sequence_score", "crf_log_norm", "crf_log_likelihood",
    "crf_unary_score", "crf_binary_score", "CrfForwardRnnCell",
    "viterbi_decode", "crf_decode", "CrfDecodeForwardRnnCell",
    "CrfDecodeBackwardRnnCell", "crf_multitag_sequence_score"
]


# 正确路径的分数，CRF的分子
def crf_sequence_score(inputs, tag_indices, sequence_lengths,
                       transition_params):
    """Computes the unnormalized score for a tag sequence.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
          compute the unnormalized score.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """

    # 当句子长度是1时，就没必要计算转移分数。直接从inputs中取出对应tag的值即可
    def _single_seq_fn():
        batch_size = array_ops.shape(inputs, out_type=tag_indices.dtype)[0]  # （2）
        example_inds = array_ops.reshape(math_ops.range(batch_size, dtype=tag_indices.dtype), [-1, 1])  # [[0], [1]]
        sequence_scores = array_ops.gather_nd(array_ops.squeeze(inputs, [1]),
                                              array_ops.concat([example_inds, tag_indices], axis=1))
        sequence_scores = array_ops.where(math_ops.less_equal(sequence_lengths, 0),
                                          array_ops.zeros_like(sequence_scores),
                                          sequence_scores)
        return sequence_scores

    # 当句子长度大于1时，就必须计算转移分数，是正常的状态
    def _multi_seq_fn():
        # Compute the scores of the given tag sequence.
        unary_scores = crf_unary_score(tag_indices, sequence_lengths, inputs)
        binary_scores = crf_binary_score(tag_indices, sequence_lengths, transition_params)
        sequence_scores = unary_scores + binary_scores
        return sequence_scores

    return utils.smart_cond(
        pred=math_ops.equal(
            tensor_shape.dimension_value(
                inputs.shape[1]) or array_ops.shape(inputs)[1],
            1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)


def crf_multitag_sequence_score(inputs, tag_bitmap, sequence_lengths,
                                transition_params):
    """Computes the unnormalized score of all tag sequences matching tag_bitmap.
    tag_bitmap enables more than one tag to be considered correct at each time
    step. This is useful when an observed output at a given time step is
    consistent with more than one tag, and thus the log likelihood of that
    observation must take into account all possible consistent tags.
    Using one-hot vectors in tag_bitmap gives results identical to
    crf_sequence_score.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_bitmap: A [batch_size, max_seq_len, num_tags] boolean tensor
          representing all active tags at each index for which to calculate the
          unnormalized score.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """

    # If max_seq_len is 1, we skip the score calculation and simply gather the
    # unary potentials of all active tags.
    def _single_seq_fn():
        filtered_inputs = array_ops.where(
            tag_bitmap, inputs,
            array_ops.fill(array_ops.shape(inputs), float("-inf")))
        return math_ops.reduce_logsumexp(
            filtered_inputs, axis=[1, 2], keepdims=False)

    def _multi_seq_fn():
        # Compute the logsumexp of all scores of sequences matching the given tags.
        filtered_inputs = array_ops.where(
            tag_bitmap, inputs,
            array_ops.fill(array_ops.shape(inputs), float("-inf")))
        return crf_log_norm(
            inputs=filtered_inputs,
            sequence_lengths=sequence_lengths,
            transition_params=transition_params)

    return utils.smart_cond(
        pred=math_ops.equal(
            tensor_shape.dimension_value(
                inputs.shape[1]) or array_ops.shape(inputs)[1],
            1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)


def crf_log_norm(inputs, sequence_lengths, transition_params):
    """Computes the normalization for a CRF.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
      log_norm: A [batch_size] vector of normalizers for a CRF.
    """
    # inputs: (2, 3, 4)
    first_input = array_ops.slice(inputs, [0, 0, 0], [-1, 1, -1])  # (2, 1, 4)
    first_input = array_ops.squeeze(first_input, [1])  # (2, 4)

    # If max_seq_len is 1, we skip the algorithm and simply reduce_logsumexp over
    # the "initial state" (the unary potentials).
    def _single_seq_fn():
        log_norm = math_ops.reduce_logsumexp(first_input, [1])  # (2,)
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = array_ops.where(math_ops.less_equal(sequence_lengths, 0),
                                   array_ops.zeros_like(log_norm),
                                   log_norm)
        return log_norm

    def _multi_seq_fn():
        """Forward computation of alpha values."""
        rest_of_input = array_ops.slice(inputs, [0, 1, 0], [-1, -1, -1])  # (2, 2, 4)

        # Compute the alpha values in the forward algorithm in order to get the
        # partition function.
        forward_cell = CrfForwardRnnCell(transition_params)
        # Sequence length is not allowed to be less than zero.
        sequence_lengths_less_one = math_ops.maximum(constant_op.constant(0, dtype=sequence_lengths.dtype),
                                                     sequence_lengths - 1)
        _, alphas = rnn.dynamic_rnn(
            cell=forward_cell,
            inputs=rest_of_input,
            sequence_length=sequence_lengths_less_one,
            initial_state=first_input,
            dtype=dtypes.float32)
        log_norm = math_ops.reduce_logsumexp(alphas, [1])
        # Mask `log_norm` of the sequences with length <= zero.
        log_norm = array_ops.where(math_ops.less_equal(sequence_lengths, 0),
                                   array_ops.zeros_like(log_norm),
                                   log_norm)
        return log_norm

    return utils.smart_cond(
        pred=math_ops.equal(
            tensor_shape.dimension_value(
                inputs.shape[1]) or array_ops.shape(inputs)[1],
            1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)


def crf_log_likelihood(inputs, tag_indices, sequence_lengths, transition_params=None):
    """Computes the log-likelihood of tag sequences in a CRF.
    Args:
      inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
          to use as input to the CRF layer.
      tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
          compute the log-likelihood.
      sequence_lengths: A [batch_size] vector of true sequence lengths.
      transition_params: A [num_tags, num_tags] transition matrix, if available.
    Returns:
      log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
        each example, given the sequence of tag indices.
      transition_params: A [num_tags, num_tags] transition matrix. This is either
          provided by the caller or created in this function.
    """
    # Get shape information.
    num_tags = tensor_shape.dimension_value(inputs.shape[2])

    # Get the transition matrix if not provided.
    if transition_params is None:
        transition_params = vs.get_variable("transitions", [num_tags, num_tags])
    # 一个tag序列的得分(状态分数 + 转移分数)，假设用s表示
    sequence_scores = crf_sequence_score(inputs, tag_indices, sequence_lengths, transition_params)
    # 所有可能tag序列的得分，并将所有得分取exp，然后相加，再取log
    log_norm = crf_log_norm(inputs, sequence_lengths, transition_params)

    # sequence_scores可以看成 log(exp(sequence_scores))，因为后面的减数实际上是log(sum(exp(any_sequence_scores)))
    # 二者相减，实际就是当前tag序列的概率值取log。这也就是 log_likelihood的含义。
    log_likelihood = sequence_scores - log_norm
    return log_likelihood, transition_params


# 一元分值，状态分数，不是概率
def crf_unary_score(tag_indices, sequence_lengths, inputs):
    """
    crf_unary_score计算一元概率，输出一个维度为[batch_size]的向量，向量中的每一个元素是一个sequence中所有的真实标签概率之和。
    inputs的维度为[batch_size, max_seq_len, num_tags]，一般为BiLSTM或者是Bert的输出。
    num_tags为标签数
    inputs[i][k]表示的是，当前batch中的第i个序列，其位置j对应的输出属于类别k的概率。
    flattened_inputs将input变为一维， tag_indices为真实的标签，维度为[batch_size, max_seq_len]
    offsets的维度为[batch_size, max_seq_len], offsetsi = (batch_size i + max_seq_len j) * num_tags
    因此，offsets + tag_indices中(i，j)位置的值表示的是，当前batch中的第i个序列，其位置j对应的输出属于类别tag_indicesi的概率位于flatten_inputs
    拉平后得到 flattened_tag_indices，通过gather函数，得到所有的一元概率
    最后，使用sequence_mask根据序列的真是长度，生成mask，得到所有有效的一元概率之和unary_scores。
    """
    # inputs: (2, 3, 4)
    batch_size = array_ops.shape(inputs)[0]  # 2
    max_seq_len = array_ops.shape(inputs)[1]  # 3
    num_tags = array_ops.shape(inputs)[2]  # 4
    # 将整个输入拉平，变成一维数组
    flattened_inputs = array_ops.reshape(inputs, [-1])  # (24,)
    # 确定铺平后，batch中每个句子的起点
    # [0, 1] * 3 * 4 -> [0, 12] -> [[0], [12]]
    offsets = array_ops.expand_dims(math_ops.range(batch_size) * max_seq_len * num_tags, 1)
    # 确定铺平后，batch中每个句子中的每个字的起点
    # [[0], [12]] + [[0, 1, 2] * 4] -> [[0, 4, 8], [12, 16, 20]]
    offsets += array_ops.expand_dims(math_ops.range(max_seq_len) * num_tags, 0)  # (2, 3)
    # Use int32 or int64 based on tag_indices' dtype.
    if tag_indices.dtype == dtypes.int64:
        offsets = math_ops.cast(offsets, dtypes.int64)
    # 将tag_indices转化成铺平后的indices
    flattened_tag_indices = array_ops.reshape(offsets + tag_indices, [-1])  # (6,)
    # 根据索引从参数轴上收集切片，并重新reshape
    # input = [[[1, 1, 1], [2, 2, 2]],
    #          [[3, 3, 3], [4, 4, 4]],
    #          [[5, 5, 5], [6, 6, 6]]]
    # tf.gather(input, [0, 2]) ==> [[[1, 1, 1], [2, 2, 2]],
    #                               [[5, 5, 5], [6, 6, 6]]]
    unary_scores = array_ops.reshape(
        array_ops.gather(flattened_inputs, flattened_tag_indices),
        [batch_size, max_seq_len])  # (2, 3)
    # 根据句子长度，生成mask
    masks = array_ops.sequence_mask(sequence_lengths,
                                    maxlen=array_ops.shape(tag_indices)[1],
                                    dtype=dtypes.float32)  # (2, 3)
    # 将每个字符的分数相加，作为标签序列的unary_scores
    unary_scores = math_ops.reduce_sum(unary_scores * masks, 1)  # (2,)
    return unary_scores


# 二元分值，转移分数，不是概率
def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    """
    crf_binary_score计算二元概率，输出维度为[batch_size]，向量中的每个元素是一个sequence中所有的转移概率之和。
    序列的长度为num_transitions，则会发生num_transitions-1此转移
    转移的起始下标start_tag_indices对应 [0, ..., num_transitions-2]
    结束下标end_tag_indices对应 [1, ..., num_transitions-1]
    使用与crf_unary_score类似的原理取出对应位置的转移概率，进行mask操作求和，返回binary_scores。
    """
    # tag_indices:(2, 3), transition_params: (4, 4)
    # Get shape information.
    num_tags = transition_params.get_shape()[0]  # 4
    # 发生转移的次数
    num_transitions = array_ops.shape(tag_indices)[1] - 1  # 2

    # tag_indices = [[4, 1, 2], [0, 2, 1]]
    # -> start_tag_indices: [[4, 1], [0, 2]]
    # -> end_tag_indices: [[1, 2], [2, 1]]
    start_tag_indices = array_ops.slice(tag_indices, [0, 0], [-1, num_transitions])  # (2, 2)
    end_tag_indices = array_ops.slice(tag_indices, [0, 1], [-1, num_transitions])  # (2, 2)

    # Encode the indices in a flattened representation.
    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = array_ops.reshape(transition_params, [-1])

    # 在flattened_transition_params中获得每一个转移的分数值
    binary_scores = array_ops.gather(flattened_transition_params, flattened_transition_indices)  # (2, 2)

    masks = array_ops.sequence_mask(sequence_lengths,
                                    maxlen=array_ops.shape(tag_indices)[1],
                                    dtype=dtypes.float32)  # (2, 3)
    truncated_masks = array_ops.slice(masks, [0, 1], [-1, -1])  # (2, 2)
    binary_scores = math_ops.reduce_sum(binary_scores * truncated_masks, 1)  # (2,)
    return binary_scores


class CrfForwardRnnCell(rnn_cell.RNNCell):
    """Computes the alpha values in a linear-chain CRF.
    See http://www.cs.columbia.edu/~mcollins/fb.pdf for reference.
    """

    def __init__(self, transition_params):
        """Initialize the CrfForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary potentials.
              This matrix is expanded into a [1, num_tags, num_tags] in preparation
              for the broadcast summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)  # (4, 4) -> (1, 4, 4)
        self._num_tags = tensor_shape.dimension_value(transition_params.shape[0])

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        """Build the CrfForwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous alpha
              values.
          scope: Unused variable scope of this cell.
        Returns:
          new_alphas, new_alphas: A pair of [batch_size, num_tags] matrices
              values containing the new alpha values.
        """
        # inputs: (2, 4), state: (2, 4)
        state = array_ops.expand_dims(state, 2)  # (2, 4, 1)
        transition_scores = state + self._transition_params  # (2, 4, 1) + (1, 4, 4) -> (2, 4, 4)
        new_alphas = inputs + math_ops.reduce_logsumexp(transition_scores, [1])  # (2, 4) + (2, 4) -> (2, 4)
        # 为何不是如下形式 ???
        # transition_scores = inputs + state + self._transition_params
        # new_alphas = math_ops.reduce_logsumexp(transition_scores, [1])
        # 答案: log_sum_exp(Emi_i_j) = Emi_i_j

        return new_alphas, new_alphas


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
          indices.
      viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()

    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score


class CrfDecodeForwardRnnCell(rnn_cell.RNNCell):
    """Computes the forward decoding in a linear-chain CRF.
    """

    def __init__(self, transition_params):
        """Initialize the CrfDecodeForwardRnnCell.
        Args:
          transition_params: A [num_tags, num_tags] matrix of binary
            potentials. This matrix is expanded into a
            [1, num_tags, num_tags] in preparation for the broadcast
            summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)
        self._num_tags = tensor_shape.dimension_value(transition_params.shape[0])

    @property
    def state_size(self):
        return self._num_tags

    @property
    def output_size(self):
        return self._num_tags

    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeForwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of unary potentials.
          state: A [batch_size, num_tags] matrix containing the previous step's
                score values.
          scope: Unused variable scope of this cell.
        Returns:
          backpointers: A [batch_size, num_tags] matrix of backpointers.
          new_state: A [batch_size, num_tags] matrix of new score values.
        """
        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
        state = array_ops.expand_dims(state, 2)  # [B, O, 1]

        # This addition op broadcasts self._transitions_params along the zeroth
        # dimension and state along the second dimension.
        # [B, O, 1] + [1, O, O] -> [B, O, O]
        transition_scores = state + self._transition_params  # [B, O, O]
        new_state = inputs + math_ops.reduce_max(transition_scores, [1])  # [B, O]
        backpointers = math_ops.argmax(transition_scores, 1)
        backpointers = math_ops.cast(backpointers, dtype=dtypes.int32)  # [B, O]
        return backpointers, new_state


class CrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
    """Computes backward decoding in a linear-chain CRF.
    """

    def __init__(self, num_tags):
        """Initialize the CrfDecodeBackwardRnnCell.
        Args:
          num_tags: An integer. The number of tags.
        """
        self._num_tags = num_tags

    @property
    def state_size(self):
        return 1

    @property
    def output_size(self):
        return 1

    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeBackwardRnnCell.
        Args:
          inputs: A [batch_size, num_tags] matrix of
                backpointer of next step (in time order).
          state: A [batch_size, 1] matrix of tag index of next step.
          scope: Unused variable scope of this cell.
        Returns:
          new_tags, new_tags: A pair of [batch_size, num_tags]
            tensors containing the new tag indices.
        """
        state = array_ops.squeeze(state, axis=[1])  # [B]
        batch_size = array_ops.shape(inputs)[0]
        b_indices = math_ops.range(batch_size)  # [B]
        indices = array_ops.stack([b_indices, state], axis=1)  # [B, 2]
        new_tags = array_ops.expand_dims(
            gen_array_ops.gather_nd(inputs, indices),  # [B]
            axis=-1)  # [B, 1]

        return new_tags, new_tags


def crf_decode(potentials, transition_params, sequence_length):
    """Decode the highest scoring sequence of tags in TensorFlow.
    This is a function for tensor.
    Args:
      potentials: A [batch_size, max_seq_len, num_tags] tensor of
                unary potentials.
      transition_params: A [num_tags, num_tags] matrix of
                binary potentials.
      sequence_length: A [batch_size] vector of true sequence lengths.
    Returns:
      decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
                  Contains the highest scoring tag indices.
      best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """

    # If max_seq_len is 1, we skip the algorithm and simply return the argmax tag
    # and the max activation.
    def _single_seq_fn():
        squeezed_potentials = array_ops.squeeze(potentials, [1])
        decode_tags = array_ops.expand_dims(
            math_ops.argmax(squeezed_potentials, axis=1), 1)
        best_score = math_ops.reduce_max(squeezed_potentials, axis=1)
        return math_ops.cast(decode_tags, dtype=dtypes.int32), best_score

    def _multi_seq_fn():
        """Decoding of highest scoring sequence."""

        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
        num_tags = tensor_shape.dimension_value(potentials.shape[2])

        # Computes forward decoding. Get last score and backpointers.
        crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params)
        initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])
        initial_state = array_ops.squeeze(initial_state, axis=[1])  # [B, O]
        inputs = array_ops.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
        # Sequence length is not allowed to be less than zero.
        sequence_length_less_one = math_ops.maximum(
            constant_op.constant(0, dtype=sequence_length.dtype),
            sequence_length - 1)
        backpointers, last_score = rnn.dynamic_rnn(  # [B, T - 1, O], [B, O]
            crf_fwd_cell,
            inputs=inputs,
            sequence_length=sequence_length_less_one,
            initial_state=initial_state,
            time_major=False,
            dtype=dtypes.int32)
        backpointers = gen_array_ops.reverse_sequence(  # [B, T - 1, O]
            backpointers, sequence_length_less_one, seq_dim=1)

        # Computes backward decoding. Extract tag indices from backpointers.
        crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
        initial_state = math_ops.cast(math_ops.argmax(last_score, axis=1),  # [B]
                                      dtype=dtypes.int32)
        initial_state = array_ops.expand_dims(initial_state, axis=-1)  # [B, 1]
        decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, 1]
            crf_bwd_cell,
            inputs=backpointers,
            sequence_length=sequence_length_less_one,
            initial_state=initial_state,
            time_major=False,
            dtype=dtypes.int32)
        decode_tags = array_ops.squeeze(decode_tags, axis=[2])  # [B, T - 1]
        decode_tags = array_ops.concat([initial_state, decode_tags],  # [B, T]
                                       axis=1)
        decode_tags = gen_array_ops.reverse_sequence(  # [B, T]
            decode_tags, sequence_length, seq_dim=1)

        best_score = math_ops.reduce_max(last_score, axis=1)  # [B]
        return decode_tags, best_score

    return utils.smart_cond(
        pred=math_ops.equal(tensor_shape.dimension_value(potentials.shape[1]) or
                            array_ops.shape(potentials)[1], 1),
        true_fn=_single_seq_fn,
        false_fn=_multi_seq_fn)
