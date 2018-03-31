python mnist_main.py --dpath ~/udata/pytorch_data/ --epochs 500 --dp-dense 0.1 --weight-decay 0.000001 --no-save | tee ~/udata/logs/regularization/test000001.log
python mnist_main.py --dpath ~/udata/pytorch_data/ --epochs 500 --dp-dense 0.1 --weight-decay 0.000002 --no-save | tee ~/udata/logs/regularization/test000002.log
python mnist_main.py --dpath ~/udata/pytorch_data/ --epochs 500 --dp-dense 0.1 --weight-decay 0.000004 --no-save | tee ~/udata/logs/regularization/test000004.log
python mnist_main.py --dpath ~/udata/pytorch_data/ --epochs 500 --dp-dense 0.1 --weight-decay 0.000006 --no-save | tee ~/udata/logs/regularization/test000006.log
python mnist_main.py --dpath ~/udata/pytorch_data/ --epochs 500 --dp-dense 0.1 --weight-decay 0.000008 --no-save | tee ~/udata/logs/regularization/test000008.log
