from dataclasses import dataclass, field

@dataclass
class RunGlueAuxiliaryArguments:
    """
    RunGlueAuxiliaryArguments is the subset of the arguments we use in the run_glue_auxiliary.py script
    """
    # additional arguments required for e-budur

    model_type: str = field(
        default=None,
        metadata={
            "help": (
                "Model type."
            )
        },
    )

    evaluation_steps: int = field(
        default=0,
        metadata={
            "help": (
                "Log every X updates steps."
            )
        },
    )

    eval_split_name: str = field(
        default='dev',
        metadata={
            "help": (
                "The name of the evaluation split."
            )
        },
    )

    dynamic_evaluation_step_regime: bool = field(
        default=False,
        metadata={
            "help": (
                "Apply dynamic evaluation regime."
            )
        },
    )

    tensorboard_log_dir: str = field(
        default=None,
        metadata={
            "help": (
                "For logging directory of tensorboard."
            )
        },
    )

    zemberek_path: str = field(
        default=None,
        metadata={
            "help": (
                "The zemberek library path."
            )
        },
    )

    java_home_path: str = field(
        default=None,
        metadata={
            "help": (
                "The java home path."
            )
        },
    )

    do_morphological_preprocessing: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to preprocess the input file using a morphological parser."
            )
        },
    )

    morphological_parser_name: str = field(
        default='zemberek',
        metadata={
            "help": (
                "The name of morphological parser."
            )
        },
    )

    omit_suffixes_after_morphological_preprocessing: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to omit suffixes in the resulting tokens of a morphological parser."
            )
        },
    )

    boun_parser_dir: str = field(
        default=None,
        metadata={
            "help": (
                "The path of boun parser."
            )
        },
    )

    boun_parser_python_path: str = field(
        default=None,
        metadata={
            "help": (
                "The path of boun parser python env.."
            )
        },
    )

    do_ngram_preprocessing: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to preprocess the input file by using ngram sequence method."
            )
        },
    )

    ngram_size: int = field(
        default=None,
        metadata={
            "help": (
                "The size of ngram when do_ngram_preprocessing is True."
            )
        },
    )

    do_sentencepiece_preprocessing: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to preprocess the input file using the sentencepiece parser."
            )
        },
    )

    sp_model_path: str = field(
        default='spm.model',
        metadata={
            "help": (
                "The path of sentencepiece model."
            )
        },
    )