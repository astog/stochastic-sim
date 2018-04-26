for ratio in 3.1 3.2 3.3 3.4 3.5 3.6 3.7 3.8 3.9 4.0
do
    python main_mnist.py --dpath ~/udata/pytorch_data/ --epochs 50 --wd 1e-6 --infl-ratio $ratio --npasses 1 --save-path ~/udata/logs/bnn/mnist/lenet/1-inf$ratio.mkl | tee ~/udata/logs/bnn/mnist/lenet/1-inf$ratio.log
done
