import tensorflow as tf

from ....utils.ner.post_process import extract_annotations_from_model_preds

from ..metrics import format_distemist_preds, format_distemist_df, \
    calculate_distemist_metrics


class EarlyDist_Val(tf.keras.callbacks.Callback):
    """
    Custom callback that performs early-stopping strategy monitoring F1 metric
    on validation dataset. P, R, F1 values are reported at the end
    of each epoch.

    TODO: not adapted to CRF classifier
    """
    def __init__(
            self, x_val, val_frag, val_doc_list, val_start_end, val_word_id,
            df_text_val, df_val_gs, valid_codes, preds_frag_tok,
            ann_extractor, word_preds_converter,
            patience=10, subtask="ner",
            logits=True, n_output=1, text_col="raw_text"
    ) -> None:
        self.X_val = x_val
        self.val_frag = val_frag
        self.val_doc_list = val_doc_list
        self.val_start_end = val_start_end
        self.val_word_id = val_word_id
        self.df_text_val = df_text_val
        self.df_val_gs = df_val_gs
        self.valid_codes = valid_codes
        self.preds_frag_tok = preds_frag_tok
        self.ann_extractor = ann_extractor
        self.word_preds_converter = word_preds_converter
        self.patience = patience
        self.subtask = subtask
        self.logits = logits
        self.n_output = n_output
        self.text_col = text_col

    def on_train_begin(self, logs=None) -> None:
        self.best = 0.0
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs={}) -> None:
        # Metrics reporting
        y_pred_val = self.model.predict(self.X_val)
        # Multiple labels
        if self.n_output == 1:  # single output tensor
            y_pred_val = [y_pred_val]

        if self.logits:
            # Multiple labels
            for lab_i in range(self.n_output):
                y_pred_val[lab_i] = tf.nn.softmax(
                    logits=y_pred_val[lab_i],
                    axis=-1
                ).numpy()

        df_pred_val = extract_annotations_from_model_preds(
            arr_doc=self.val_doc_list, arr_frags=self.val_frag,
            arr_preds=y_pred_val, arr_start_end=self.val_start_end,
            arr_word_id=self.val_word_id,
            arr_preds_pos_tok=self.preds_frag_tok.calculate_pos_tok(
                arr_len=self.val_start_end
            ),
            ann_extractor=self.ann_extractor,
            word_preds_converter=self.word_preds_converter
        )

        # To avoid errors: in the first epochs, the predicted codes
        # are not valid, so ignore those predictions
        if df_pred_val.shape[0] == 0:
            print("Corrupted val predictions!")
            p_val, r_val, f1_val = .0, .0, .0
        else:
            # Format predictions
            df_pred_val = format_distemist_preds(
                df_preds=df_pred_val,
                df_text=self.df_text_val,
                text_col=self.text_col
            )

            df_pred_val = format_distemist_df(
                df=df_pred_val,
                valid_codes=self.valid_codes
            )

            # Micro-avg
            p_val, r_val, f1_val = calculate_distemist_metrics(
                gs=self.df_val_gs,
                pred=df_pred_val,
                subtask=self.subtask
            )
            p_val = round(p_val, 4)
            r_val = round(r_val, 4)
            f1_val = round(f1_val, 4)
        logs['p_val'] = p_val
        logs['r_val'] = r_val
        logs['f1_val'] = f1_val

        print('\rval_p: %s - val_r: %s - val_f1: %s' %
              (str(p_val), str(r_val), str(f1_val)), end=100*' '+'\n')

        # Early-stopping
        if (f1_val > self.best):
            self.best = f1_val
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None) -> None:
        self.model.set_weights(self.best_weights)
