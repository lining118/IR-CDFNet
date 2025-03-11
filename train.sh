#!/bin/bash
# Run script
# DIS/COD/HRSOD: epochs,val_last,step:[500,200,10]/[150,50,10]/[150,50,10]
method="$1"
epochs=80
resume_epoches=32
val_last=10
step=10
testsets=NO     # Non-existing folder to skip.
# testsets=TE-COD10K   # for COD

# Train
devices=$2
nproc_per_node=$(echo ${devices%%,} | grep -o "," | wc -l)

to_be_distributed=`echo ${nproc_per_node} | awk '{if($e > 0) print "True"; else print "False";}'`

echo Training started at $(date)
if [ ${to_be_distributed} == "True" ]
then
    # Adapt the nproc_per_node by the number of GPUs. Give 29500 as the default value of master_port.
    echo "Multi-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    torchrun --nproc_per_node $((nproc_per_node+1)) --master_port=$((29500+${3:-11})) \
    train.py --ckpt_dir ckpt/${method} --epochs ${epochs} \
        --testsets ${testsets} \
        --dist ${to_be_distributed}
else
    echo "Single-GPU mode received..."
    CUDA_VISIBLE_DEVICES=${devices} \
    python train.py --ckpt_dir ckpt/${method} --epochs ${epochs}\
        --testsets ${testsets} \
        --dist ${to_be_distributed} \
        --resume ckpt/${method}/ep${resume_epoches}.pth
fi

echo Training finished at $(date)
