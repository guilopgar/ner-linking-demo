import tensorflow as tf


class TokenClassificationLoss(tf.keras.losses.Loss):
    """
    Code adapted from
    https://huggingface.co/transformers/_modules/transformers/modeling_tf_utils.html#TFTokenClassificationLoss
    """
    def __init__(
            self, from_logits=True, ignore_val=-100,
            reduction=tf.keras.losses.Reduction.AUTO, **kwargs):
        self.from_logits = from_logits
        self.ignore_val = ignore_val
        self.reduction = reduction
        super(TokenClassificationLoss, self).__init__(**kwargs)

    def call(self, y_true, y_pred, sample_weight=None):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits, reduction=self.reduction
        )
        # make sure only labels that are not equal to self.ignore_val
        # are taken into account as loss
        active_loss = tf.reshape(y_true, (-1,)) != self.ignore_val
        reduced_preds = tf.boolean_mask(
            tf.reshape(
                y_pred, (-1, y_pred.shape[2])
            ),
            active_loss
        )
        labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)

        return loss_fn(labels, reduced_preds, sample_weight=sample_weight)


class TokenClassificationLossSampleWeight(tf.keras.losses.Loss):
    """
    Code adapted from
    https://huggingface.co/transformers/_modules/transformers/modeling_tf_utils.html#TFTokenClassificationLoss
    """
    def __init__(self, weak_label, weak_weight_value=1,
                 strong_weight_value=2, from_logits=True,
                 ignore_val=-100, **kwargs):
        self.weak_label = weak_label
        self.weak_weight_value = weak_weight_value
        self.strong_weight_value = strong_weight_value
        self.from_logits = from_logits
        self.ignore_val = ignore_val
        super(TokenClassificationLossSampleWeight, self).__init__(**kwargs)

    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=self.from_logits,
            reduction=tf.keras.losses.Reduction.NONE
        )
        # make sure only labels that are not equal to self.ignore_val
        # are taken into account as loss
        active_loss = tf.reshape(y_true, (-1,)) != self.ignore_val
        reduced_preds = tf.boolean_mask(
            tf.reshape(
                y_pred, (-1, y_pred.shape[2])
            ),
            active_loss
        )
        labels = tf.boolean_mask(tf.reshape(y_true, (-1,)), active_loss)

        # sample weight
        sample_weight = tf.where(
            labels == self.weak_label,
            x=self.weak_weight_value,
            y=self.strong_weight_value
        )
        loss_value = loss_fn(labels, reduced_preds)

        return loss_value * tf.cast(sample_weight, loss_value.dtype)
