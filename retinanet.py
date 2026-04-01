from manga_detection.training.runner import main


if __name__ == "__main__":
    main(
        default_model="retinanet",
        default_epochs=15,
        default_batch_size=4,
        default_lr=1e-4,
        default_output_name="retinanet_pretrained_augment_lr=0.0001_imgaug",
    )
