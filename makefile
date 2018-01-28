train:
	python main.py \
		--phase train \
		--arch mike --edge_factor 0.2 --l2_factor 0.2 \
		--name N-l2.2-e.2

eval:
	python main.py \
		--phase eval \
		--arch mike --pretrain_path output/checkpoint/M-mike-l2.2/net-19.pth \
		--name D-mike-l2.2 --tri_visual

eval_search:
	python main.py \
		--phase eval_search \
		--arch mike --pretrain_path output/checkpoint/M-mike-l1.2/
