python3 -u main.py \
  --data_root "data/ModelNet10" \
  --epochs 250 \
  --batch_size 32 \
  --npoints 4096 \
  --lr 0.01 \
  --num_workers 4 \
  --performer=True \
  --performer_nb_features 32 \
  --performer_redraw_interval 1000 \
  --save_path "checkpoints/PCT_performer_best.pt" \
  --log_path "logs/performer.log" 


python3 -u main.py \
  --data_root "/root/autodl-tmp/ModelNet40" \
  --epochs 250 \
  --batch_size 32 \
  --npoints 1024 \
  --lr 0.01 \
  --num_workers 4 \
  --performer=True \
  --performer_nb_features 32 \
  --performer_redraw_interval 1000 \
  --off_new_format True \
  --save_path "checkpoints/PCT_performer40_best.pt" \
  --log_path "logs/performer40.log" 

python3 -u main.py \
   --data_root "/root/autodl-tmp/ModelNet40" \
  --epochs 250 \
  --batch_size 32 \
  --npoints 1024 \
  --lr 0.01 \
  --num_workers 4 \
  --off_new_format True \
  --save_path "checkpoints/PCT40_best.pt" \
  --log_path "logs/baseline40.log"


python3 -u main.py \
  --data_root "/root/autodl-tmp/ModelNet40" \
  --epochs 250 \
  --batch_size 32 \
  --npoints 1024 \
  --lr 0.01 \
  --num_workers 4 \
  --performer=True \
  --performer_nb_features 32 \
  --performer_redraw_interval 1000 \
  --off_new_format True \
  --save_path "checkpoints/PCT_performer_dist40_best.pt" \
  --log_path "logs/performer_dist40.log" \
  --add_dist=True

python3 -u main.py \
  --data_root "/root/autodl-tmp/ModelNet40" \
  --epochs 250 \
  --batch_size 32 \
  --npoints 4096 \
  --lr 0.01 \
  --num_workers 4 \
  --performer=True \
  --performer_nb_features 32 \
  --performer_redraw_interval 1000 \
  --off_new_format True \
  --save_path "checkpoints/PCT_performer4096_best.pt" \
  --log_path "logs/performer4096.log" 

python3 -u main.py \
   --data_root "/root/autodl-tmp/ModelNet40" \
  --epochs 250 \
  --batch_size 32 \
  --npoints 4096 \
  --lr 0.01 \
  --num_workers 4 \
  --off_new_format True \
  --save_path "checkpoints/PCT4096_best.pt" \
  --log_path "logs/baseline4096.log"