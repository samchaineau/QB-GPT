import tensorflow as tf

class CustomSparseCategoricalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, from_logits=False, class_weights=None):
        super(CustomSparseCategoricalCrossentropy, self).__init__()
        self.from_logits = from_logits
        self.class_weights = class_weights

    def call(self, y_true, y_pred):
        mask = tf.not_equal(y_true, -100)  # Create a mask for valid labels

        if self.from_logits == True:
          valid_preds = tf.nn.softmax(y_pred)
        else:
          valid_preds = y_pred

        valid_labels = tf.boolean_mask(y_true, mask)
        valid_logits = tf.boolean_mask(valid_preds, mask)

        # Apply class weights if provided
        if self.class_weights is not None:
            # Create a tensor of weights using tf.gather
            weights = tf.gather(tf.constant(list(self.class_weights.values()), dtype=tf.float32), tf.cast(valid_labels, tf.int32))
            weighted_loss = tf.keras.losses.sparse_categorical_crossentropy(valid_labels, valid_logits)
            weighted_loss = weighted_loss * weights
            loss = tf.reduce_mean(weighted_loss)
        else:
            loss = tf.keras.losses.sparse_categorical_crossentropy(valid_labels, valid_logits)

        return loss

class CustomSparseCategoricalAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='custom_sparse_categorical_accuracy', **kwargs):
        super(CustomSparseCategoricalAccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, -100)  # Create a mask for valid labels
        valid_labels = tf.boolean_mask(y_true, mask)

        preds = tf.nn.softmax(y_pred)
        preds = tf.argmax(preds, axis = -1)
        valid_preds = tf.boolean_mask(preds, mask)

        correct = tf.equal(valid_labels, valid_preds)

        accuracy = tf.reduce_mean(tf.cast(correct, dtype=tf.float32))

        self.total.assign_add(accuracy)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count if self.count > 0 else 0.0

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)

class CustomTopKAccuracy(tf.keras.metrics.Metric):
    def __init__(self, k=3, name='custom_top_k_accuracy', **kwargs):
        super(CustomTopKAccuracy, self).__init__(name=name, **kwargs)
        self.k = k
        self.total = self.add_weight(name='total', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.not_equal(y_true, -100)  # Create a mask for valid labels
        valid_labels = tf.boolean_mask(y_true, mask)

        # Get top-k predicted classes
        preds = tf.nn.softmax(y_pred)
        top_k_values, top_k_indices = tf.nn.top_k(preds, k=self.k)
        valid_preds = tf.boolean_mask(top_k_indices, mask)

        # Broadcast valid_labels to match the shape of valid_preds
        valid_labels_broadcasted = tf.tile(tf.expand_dims(valid_labels, axis=-1), [1, self.k])

        valid_labels_broadcasted = tf.cast(valid_labels_broadcasted, dtype=tf.int32)
        valid_preds = tf.cast(valid_preds, dtype=tf.int32)

        correct = tf.reduce_sum(tf.cast(tf.equal(valid_labels_broadcasted, valid_preds), dtype=tf.float32))

        accuracy = correct / tf.cast(tf.shape(valid_labels_broadcasted)[0], dtype=tf.float32)

        self.total.assign_add(accuracy)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count if self.count > 0 else 0.0

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)