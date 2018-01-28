train:
	python main.py \
		--phase train \
		--datafold 1 --batch_size 16 --worker 8 \
		--arch mike \
		--edge_factor 0.2 --l2_factor 0.2 --name N-l2.2-e.2

eval:
	python main.py \
		--phase eval \
		--datafold 1 --batch_size 16 --worker 8 \
		--arch mike --pretrain_path output/checkpoint/M-mike-l2.2/net-19.pth --name D-mike-l2.2 --tri_visual

eval_search:
	python main.py \
		--phase eval_search \
		--datafold 1 --batch_size 16 --worker 8 \
		--arch mike --pretrain_path output/checkpoint/M-mike-l1.2/
