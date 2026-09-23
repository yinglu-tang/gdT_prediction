"""Console entry points that keep Python result objects out of exit statuses."""


def prediction_main():
    from .pipeline_prediction import main

    main()


def embedding_main():
    from .pipeline_embedding import main

    main()


def probert_main():
    from .pipeline_probert import main

    main()

