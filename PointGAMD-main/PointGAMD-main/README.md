create a conda enviorment using 

conda create -n pointgamd python=3.9

conda activate pointgamd

pip install -r requirements.txt

1) Data Preparation
Download the dataset as given in the link below

https://github.com/antao97/PointCloudDatasets/tree/master 

and put it in the following directory 'dataset/h5/'

and for ScanObjectNN Download from

https://github.com/hkust-vgd/scanobjectnn

and put it in 'dataset/h5/ScanObjectNN'

2) run the script under classification directory in terminal as below

a) ScanObjectNN

python train.py --run_name ' ' --num_points 2048 --device 'gpu' --k_n 20 --nepochs 300 --batch_size 32 --lr 0.1 --data_root '../dataset/h5/ShapeObjectNN' --dataset 'shapeobj' 

b) ModelNet40

python train.py --run_name ' ' --num_points 1024 --device 'gpu' --k_n 20 --nepochs 300 --batch_size 32 --lr 0.1 --data_root '../dataset/h5' --dataset 'modelnet40' 

3) Run the script under part_segmentation directory in terminal as below
   
python train_seg.py --run_name ' ' --num_points 2048 --device 'gpu' --k_n 24 --nepochs 300 --batch_size 32 --lr 0.1 --data_root '../dataset/h5'

