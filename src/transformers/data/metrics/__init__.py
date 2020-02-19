# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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

try:
    from scipy.stats import pearsonr, spearmanr
    from sklearn.metrics import matthews_corrcoef, f1_score, classification_report

    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn


if _has_sklearn:

    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds)
        return {
            "acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
        }

    def acc_for_nlu(preds, labels):
        intent_acc = simple_accuracy(preds['intents'], labels['intents'])
        enumerable_entities_acc = simple_accuracy(preds['enumerable_entities'], labels['enumerable_entities'])
        non_enumerable_entities_acc = simple_accuracy(preds['non_enumerable_entities'],
                                                      labels['non_enumerable_entities'])

        acc_for_nlu_result = {
                'intents': intent_acc,
                'enumerable_entities': enumerable_entities_acc,
                'non_enumerable_entities': non_enumerable_entities_acc
            }

        return acc_for_nlu_result

    def f1_for_nlu(preds, labels, average):
        intent_f1 = f1_score(labels['intents'], preds['intents'], average=average)
        enumerable_entities_f1 = f1_score(labels['enumerable_entities'], preds['enumerable_entities'], average=average)
        non_enumerable_entities_f1 = f1_score(labels['non_enumerable_entities'].flatten(),preds['non_enumerable_entities'].flatten(), average=average)

        f1_for_nlu_result = {
                'intents': intent_f1,
                'enumerable_entities': enumerable_entities_f1,
                'non_enumerable_entities': non_enumerable_entities_f1
            }

        return f1_for_nlu_result


    def simple_classification_report(preds, labels, target_names):
        simple_classification_report_result=classification_report(y_true=labels,
                                                                  y_pred=preds,
                                                                  labels=range(len(target_names)),
                                                                  target_names=target_names)
        return simple_classification_report_result

    def classification_report_for_nlu(preds, labels, target_names):
        intents_classification_report_result = simple_classification_report(labels=labels['intents'],
                                                                            preds=preds['intents'],
                                                                            target_names=target_names['intents'])
        enumerable_entities_classification_report_result = simple_classification_report(labels=labels['enumerable_entities'],
                                                                                        preds=preds['enumerable_entities'],
                                                                            target_names=target_names['enumerable_entities'])
        non_enumerable_entities_classification_report_result = simple_classification_report(labels=labels['non_enumerable_entities'].flatten(),
                                                                                            preds=preds['non_enumerable_entities'].flatten(),
                                                                            target_names=target_names['non_enumerable_entities'])

        classification_report_for_nlu_result = {
                'intents': intents_classification_report_result,
                'enumerable_entities': enumerable_entities_classification_report_result,
                'non_enumerable_entities': non_enumerable_entities_classification_report_result
            }

        return classification_report_for_nlu_result

    def acc_and_f1_and_classification_report_for_nlu(preds, labels, target_names):

        acc_for_nlu_result =  acc_for_nlu(preds, labels)
        macro_f1_for_nlu_result = f1_for_nlu(preds, labels, average='macro')
        micro_f1_for_nlu_result = f1_for_nlu(preds, labels, average='micro')
        weighted_f1_for_nlu_result = f1_for_nlu(preds, labels, average='weighted')
        classification_report_for_nlu_result = classification_report_for_nlu(preds, labels, target_names)

        acc_and_classification_report_for_nlu_result = {
            "acc": acc_for_nlu_result,
            "f1": {
                'macro':macro_f1_for_nlu_result,
                'micro': micro_f1_for_nlu_result,
                'weighted': weighted_f1_for_nlu_result
            },
            "classification_report": classification_report_for_nlu_result
        }

        return acc_and_classification_report_for_nlu_result

    def pearson_and_spearman(preds, labels):
        pearson_corr = pearsonr(preds, labels)[0]
        spearman_corr = spearmanr(preds, labels)[0]
        return {
            "pearson": pearson_corr,
            "spearmanr": spearman_corr,
            "corr": (pearson_corr + spearman_corr) / 2,
        }


    def nlu_compute_metrics(task_name, preds, labels, target_names):

        if task_name == "google-simulated-dialogue":
            return acc_and_f1_and_classification_report_for_nlu(preds, labels, target_names)
        else:
            raise KeyError(task_name)

    def glue_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "cola":
            return {"mcc": matthews_corrcoef(labels, preds)}
        elif task_name == "sst-2":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mrpc":
            return acc_and_f1(preds, labels)
        elif task_name == "sts-b":
            return pearson_and_spearman(preds, labels)
        elif task_name == "qqp":
            return acc_and_f1(preds, labels)
        elif task_name == "mnli" or task_name == 'mnli-nmt-amzn-tr':
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "mnli-mm" or task_name == 'mnli-mm-nmt-amzn-tr':
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "qnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "rte":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "wnli":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "snli" or task_name == "snli-nmt-amzn-tr":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "xnli" or task_name == "xnli-nmt-amzn-tr":
            return {"acc": simple_accuracy(preds, labels)}
        elif task_name == "hans":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)

    def xnli_compute_metrics(task_name, preds, labels):
        assert len(preds) == len(labels)
        if task_name == "xnli":
            return {"acc": simple_accuracy(preds, labels)}
        else:
            raise KeyError(task_name)
