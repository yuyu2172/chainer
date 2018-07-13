import argparse

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import training
from chainer.training import extensions
from chainer.training import triggers

from chainer.datasets import get_cifar10
from chainer.datasets import get_cifar100

import models.VGG

import numpy as np
from chainer import reporter
from chainer.functions import accuracy as accuracy_func


class Classifier(chainer.Chain):

    def __init__(self, model, loss):
        super(Classifier, self).__init__()
        with self.init_scope():
            self.model = model
        self.loss = loss

    def __call__(self, x, t):
        y = self.model(x)
        if self.loss == 'softmax':
            loss = F.softmax_cross_entropy(y, t)
            prob = F.softmax(y).data
            max_prob = self.xp.max(prob, axis=1)
            N = len(x)

            count_0 = (max_prob < 0.1).sum() / N
            count_1 = self.xp.logical_and(max_prob > 0.1, max_prob < 0.2).sum() / N
            count_2 = self.xp.logical_and(max_prob > 0.2, max_prob < 0.3).sum() / N
            count_3 = self.xp.logical_and(max_prob > 0.3, max_prob < 0.4).sum() / N
            count_4 = self.xp.logical_and(max_prob > 0.4, max_prob < 0.5).sum() / N
            count_5 = self.xp.logical_and(max_prob > 0.5, max_prob < 0.6).sum() / N
            count_6 = self.xp.logical_and(max_prob > 0.6, max_prob < 0.7).sum() / N
            count_7 = self.xp.logical_and(max_prob > 0.7, max_prob < 0.8).sum() / N
            count_8 = self.xp.logical_and(max_prob > 0.8, max_prob < 0.9).sum() / N
            count_9 = self.xp.logical_and(max_prob > 0.9, max_prob < 1.01).sum() / N
            reporter.report({'count_0': count_0}, self)
            reporter.report({'count_1': count_1}, self)
            reporter.report({'count_2': count_2}, self)
            reporter.report({'count_3': count_3}, self)
            reporter.report({'count_4': count_4}, self)
            reporter.report({'count_5': count_5}, self)
            reporter.report({'count_6': count_6}, self)
            reporter.report({'count_7': count_7}, self)
            reporter.report({'count_8': count_8}, self)
            reporter.report({'count_9': count_9}, self)
        elif self.loss == 'sigmoid':
            n_class = y.shape[1]
            new_t = self.xp.zeros((len(t), n_class), dtype=np.int32)
            new_t[self.xp.arange(len(t)), t] = 1
            loss = F.sigmoid_cross_entropy(y, new_t)

        reporter.report({'loss': loss}, self)
        accuracy = accuracy_func(y, t)
        reporter.report({'accuracy': accuracy}, self)
        return loss


def main():
    parser = argparse.ArgumentParser(description='Chainer CIFAR example:')
    parser.add_argument('--dataset', '-d', default='cifar10',
                        help='The dataset to use: cifar10 or cifar100')
    parser.add_argument('--batchsize', '-b', type=int, default=64,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=0.05,
                        help='Learning rate for SGD')
    parser.add_argument('--epoch', '-e', type=int, default=300,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--early-stopping', type=str,
                        help='Metric to watch for early stopping')
    parser.add_argument('--loss', type=str)
    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if args.dataset == 'cifar10':
        print('Using CIFAR10 dataset.')
        class_labels = 10
        train, test = get_cifar10()
    elif args.dataset == 'cifar100':
        print('Using CIFAR100 dataset.')
        class_labels = 100
        train, test = get_cifar100()
    else:
        raise RuntimeError('Invalid dataset choice.')
    model = Classifier(models.VGG.VGG(class_labels), args.loss)
    if args.gpu >= 0:
        # Make a specified GPU current
        chainer.backends.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()  # Copy the model to the GPU

    optimizer = chainer.optimizers.MomentumSGD(args.learnrate)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer_hooks.WeightDecay(5e-4))

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    stop_trigger = (args.epoch, 'epoch')
    # Early stopping option
    if args.early_stopping:
        stop_trigger = triggers.EarlyStoppingTrigger(
            monitor=args.early_stopping, verbose=True,
            max_trigger=(args.epoch, 'epoch'))

    # Set up a trainer
    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(updater, stop_trigger, out=args.out)

    # Evaluate the model with the test dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # Reduce the learning rate by half every 25 epochs.
    trainer.extend(extensions.ExponentialShift('lr', 0.5),
                   trigger=(25, 'epoch'))

    # Dump a computational graph from 'loss' variable at the first iteration
    # The "main" refers to the target link of the "main" optimizer.
    trainer.extend(extensions.dump_graph('main/loss'))

    # Take a snapshot at each epoch
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    # Write a log of evaluation statistics for each epoch
    trainer.extend(extensions.LogReport())

    # Print selected entries of the log to stdout
    # Here "main" refers to the target link of the "main" optimizer again, and
    # "validation" refers to the default name of the Evaluator extension.
    # Entries other than 'epoch' are reported by the Classifier link, called by
    # either the updater or the evaluator.
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time',
         'main/count_0',
         'main/count_1',
         'main/count_2',
         'main/count_3',
         'main/count_4',
         'main/count_5',
         'main/count_6',
         'main/count_7',
         'main/count_8',
         'main/count_9',
         ]))

    # Print a progress bar to stdout
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        # Resume from a snapshot
        chainer.serializers.load_npz(args.resume, trainer)

    # Run the training
    trainer.run()


if __name__ == '__main__':
    main()
