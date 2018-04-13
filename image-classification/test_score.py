# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""
test pretrained models
"""
from __future__ import print_function
import mxnet as mx
from common import find_mxnet, modelzoo
from score import score,score_with_thresh
import argparse

def test_models_with_threshold(args,**kwargs):
    
    (speed,r) = score_with_thresh(threshold = args.thrshold , load_epoch = args.load_epoch,image_shape = args.image_shape,model=args.pretrained_model, data_val=args.test_rec,rgb_mean='123.68,116.779,103.939', **kwargs)
    print('Tested %s, acc = %f, speed = %f img/sec' % (args.pretrained_model, r, speed))

def test_models(args,**kwargs):
    acc = mx.metric.create('acc')
    (speed,) = score(load_epoch = args.load_epoch,image_shape = args.image_shape,model=args.pretrained_model, data_val=args.test_rec,rgb_mean='123.68,116.779,103.939', metrics=acc, **kwargs)
    r = acc.get()[1]
    print('Tested %s, acc = %f, speed = %f img/sec' % (args.pretrained_model, r, speed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--test-rec', type=str,
                        help='the testrec file')
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--load-epoch', type=int,
                        help='the test epoch')
    parser.add_argument('--image-shape', type=str,
                        help='the test imageshape')
    parser.add_argument('--thrshold', type=float,
                        help='the thresh of class')
    args = parser.parse_args()
    gpus = mx.test_utils.list_gpus()
    assert len(gpus) > 0
    batch_size = 16 * len(gpus)
    gpus = ','.join([str(i) for i in gpus])
    gpus = '0'
    kwargs = {'gpus':gpus, 'batch_size':batch_size, 'max_num_examples':500}
#    download_data()
    test_models(args,**kwargs)
#    test_models_with_threshold(args,**kwargs)
