.PHONY: train predict

train:
	poetry run python -m perturb_gym.training train_from_config_file --config_file_id_or_path=multi_cell_line_custom_lpm

predict:
	poetry run python -m perturb_gym.predict --config_file multi_cell_line_custom_lpm --prediction_context HumanCellLine_CRISPRi_ARCChallenge_Val --output_dir ./predictions --model_seed 13