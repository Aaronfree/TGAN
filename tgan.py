#https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# A supporting script to make tensor layer
from tnsr import *

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from util import normal_init, get_number_parameters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Size of the batch
mb_size = 100

# Dimension of the prior
#z_dim = 10

# Size of the hidden layer
h_dim = 35

# Size of latent dim
latent_dim = 15

# Learning Rate
lr = 5e-3

# The input X
x_hat = tf.placeholder(tf.float32, shape=[None,28, 28], name='input_img')
x = tf.placeholder(tf.float32, shape=[None,28, 28], name='target_img')



def sample_z(shape):
    return np.random.uniform(-1., 1., size=shape)


#Encoder
E_U_00 = tf.Variable(normal_init([h_dim, 28]))
E_U_01 = tf.Variable(normal_init([h_dim, 28]))
E_b_0 = tf.Variable(tf.zeros(shape=[h_dim,h_dim]))
E_U_10 = tf.Variable(normal_init([latent_dim, h_dim]))
E_U_11 = tf.Variable(normal_init([latent_dim, h_dim]))
E_b_1 = tf.Variable(tf.zeros(shape=[latent_dim,latent_dim]))
def encoder(x):
    out = tensor_layer(x, [E_U_00, E_U_01], E_b_0, tf.nn.relu)
    out = tensor_layer(out, [E_U_10, E_U_11], E_b_1, tf.nn.relu)
    return out

theta_E = [E_U_00, E_U_01, E_U_10, E_U_11, E_b_0, E_b_1]    # Parameters for generator


#Generator
G_U_00 = tf.Variable(normal_init([h_dim, latent_dim]))
G_U_01 = tf.Variable(normal_init([h_dim, latent_dim]))
G_b_0 = tf.Variable(tf.zeros(shape=[h_dim,h_dim]))
G_U_10 = tf.Variable(normal_init([28, h_dim]))
G_U_11 = tf.Variable(normal_init([28, h_dim]))
G_b_1 = tf.Variable(tf.zeros(shape=[28,28]))
def generator(z):
    out = tensor_layer(z, [G_U_00, G_U_01], G_b_0, tf.nn.relu)
    out = tensor_layer(out, [G_U_10, G_U_11], G_b_1, tf.nn.sigmoid)
    return out

theta_G = [G_U_00, G_U_01, G_U_10, G_U_11, G_b_0, G_b_1]    # Parameters for generator


#Discriminator
D_U_00 = tf.Variable(normal_init([h_dim, latent_dim ]))
D_U_01 = tf.Variable(normal_init([h_dim, latent_dim ]))
D_b_0 = tf.Variable(tf.zeros(shape=[h_dim,h_dim]))
D_U_10 = tf.Variable(normal_init([1, h_dim]))
D_U_11 = tf.Variable(normal_init([1, h_dim]))
D_b_1 = tf.Variable(tf.zeros(shape=[1,1]))
def discriminator(x):
    out = tensor_layer(x, [D_U_00, D_U_01], D_b_0, tf.nn.relu)
    return tensor_layer(out, [D_U_10, D_U_11], D_b_1, tf.nn.sigmoid), tensor_layer(out, [D_U_10, D_U_11], D_b_1, identity)

theta_D = [D_U_00, D_U_01, D_U_10, D_U_11, D_b_0, D_b_1]    # Parameters for discriminator


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

print("Total number of parameters: {}".format(get_number_parameters(theta_E+theta_G+theta_D)))

#z = tf.placeholder(tf.float32, shape=[None, latent_dim ,latent_dim ], name='latent_variable')
## Reconstruction Loss
# encoding
z = encoder(x_hat)
# decoding
y = generator(z)
# loss
marginal_likelihood = -tf.reduce_mean(tf.reduce_mean(tf.squared_difference(x,y)))
theta_AE = [E_U_00, E_U_01, E_U_10, E_U_11, E_b_0, E_b_1, G_U_00, G_U_01, G_U_10, G_U_11, G_b_0, G_b_1]

# GAN Loss

z_real = tf.placeholder(tf.float32, shape=[None, latent_dim ,latent_dim ], name='prior_sample')

z_fake = z
D_real, D_real_logits = discriminator(z_real)
D_fake, D_fake_logits = discriminator(z_fake)


# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------
D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real_logits)))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake_logits)))
D_loss = D_loss_real + D_loss_fake

G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake_logits)))

neg_marginal_likelihood = - marginal_likelihood


AE_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(neg_marginal_likelihood, var_list=theta_AE)
D_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5).minimize(G_loss, var_list=theta_G)

mnist = input_data.read_data_sets('../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

for it in range(1000000):
    X_mb, _ = mnist.train.next_batch(mb_size)

    if it % 1000 == 0:
        samples = sess.run(y, feed_dict={x_hat: X_mb.reshape(mb_size, 28, 28), z_real: sample_z([mb_size, latent_dim, latent_dim])})

        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)





    _, loss_likelihood = sess.run([AE_solver, neg_marginal_likelihood],feed_dict={x_hat: X_mb.reshape(mb_size, 28, 28),x: X_mb.reshape(mb_size, 28, 28),z_real: sample_z([mb_size, latent_dim, latent_dim]) })
    _, d_loss = sess.run([D_solver, D_loss],feed_dict={x_hat: X_mb.reshape(mb_size, 28, 28),x: X_mb.reshape(mb_size, 28, 28),z_real: sample_z([mb_size, latent_dim, latent_dim]) })
    _, g_loss = sess.run([G_solver, G_loss], feed_dict={x_hat: X_mb.reshape(mb_size, 28, 28),x: X_mb.reshape(mb_size, 28, 28),z_real: sample_z([mb_size, latent_dim, latent_dim]) })
    tot_loss = loss_likelihood + d_loss + g_loss

    if it % 1000 == 0:
        print('Iter: {}'.format(it))
        print('AE loss: {:.4}'.format(loss_likelihood))
        print('D loss: {:.4}'. format(d_loss))
        print('G_loss: {:.4}'.format(d_loss))
        print('total_loss: {:.4}'.format(tot_loss))
        print('')
