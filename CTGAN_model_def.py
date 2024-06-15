import tensorflow as tf
from tensorflow import keras
class CTGAN(keras.Model):
  def __init__(self, generator, critic, latent_dim, critic_extra_steps, gp_weight):
    super().__init__()
    self.generator = generator
    self.critic = critic
    self.latent_dim = latent_dim
    self.critic_extra_steps = critic_extra_steps
    self.gp_weight = gp_weight

  def compile(self, g_opt, c_opt, c_loss_fn, g_loss_fn):
    super().compile()
    self.g_opt = g_opt
    self.c_opt = c_opt
    self.c_loss_fn = c_loss_fn
    self.g_loss_fn = g_loss_fn

  def reg_term(self, batch_size, z1, z2, c):
    """
    This function calculates the mode seeking regularisation term which reduces the euclidian distance between
    two latent vectors in the latent space created by the generator network
    """
    del_z = tf.norm(z1 - z2, ord='euclidean', axis=-1)
    del_z = tf.reduce_mean(del_z)
    [X1, c1] = self.generator([z1, c], training=True)
    [X2, c2] = self.generator([z2, c], training=True)
    del_X = tf.norm(X1 - X2, ord='euclidean', axis=-1)
    del_X = tf.reduce_mean(del_X)
    return del_X/del_z

  def gradient_penalty(self, batch_size, real_data, fake_data, one_hot_labels):
    """
    This function return the classic gradient penalty term used in WGAN networks which is used to regularize the critic
    network output
    """
    alpha = tf.random.normal([batch_size, 1], 0.0, 1.0)
    fake_data = tf.cast(fake_data, dtype = tf.float32)
    real_data = tf.cast(real_data, dtype = tf.float32)
    diff = fake_data - real_data
    interpolated_data = real_data + alpha * diff

    with tf.GradientTape() as gp_tape:
      gp_tape.watch(interpolated_data)
      pred = self.critic([interpolated_data, one_hot_labels], training=True)

    grads = gp_tape.gradient(pred, [interpolated_data])[0]
    #Calculate the norm of the gradients.
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
    return gp

  def train_step(self, data):
    real_tensor, real_labels = data
    latent_vector_size = tf.shape(real_tensor)[0]
    coeff = 1/latent_vector_size
    coeff = tf.cast(coeff, dtype=tf.dtypes.float32)
    for i in range(self.critic_extra_steps):
      random_latent_vector = tf.random.normal([latent_vector_size, self.latent_dim], 0.0, 1.0)
      with tf.GradientTape() as tape:
        [fake_data, fake_labels] = self.generator([random_latent_vector, real_labels], training = True)
        fake_logits = self.critic([fake_data, fake_labels], training = True)
        real_logits = self.critic([real_tensor, real_labels], training = True)
        c_wass_loss = self.c_loss_fn(real_logits, fake_logits)
        c_gp_loss = self.gradient_penalty(latent_vector_size, real_tensor, fake_data, fake_labels)
        c_loss = c_wass_loss + self.gp_weight * c_gp_loss
        c_loss = c_loss * coeff

      c_gradients = tape.gradient(c_loss, self.critic.trainable_variables)
      self.c_opt.apply_gradients(zip(c_gradients, self.critic.trainable_variables))

    random_latent_vector = tf.random.normal([latent_vector_size, self.latent_dim], 0.0, 1.0)
    with tf.GradientTape() as tape:
      [fake_data, fake_labels] = self.generator([random_latent_vector, real_labels], training = True)
      fake_logits = self.critic([fake_data, fake_labels])
      g_loss1 = self.g_loss_fn(fake_logits)
      g_loss2 = keras.losses.categorical_crossentropy(fake_labels, real_labels)
      z1 = tf.random.normal([latent_vector_size, self.latent_dim], 0.0, 1.0)
      z2 = tf.random.normal([latent_vector_size, self.latent_dim], 0.0, 1.0)
      mode_seeking_term = self.reg_term(latent_vector_size, z1, z2, real_labels)
      g_loss = g_loss1 + tf.reduce_mean(g_loss2) - 10.0 * mode_seeking_term
      g_loss = g_loss * coeff

    g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
    self.g_opt.apply_gradients(zip(g_gradients, self.generator.trainable_variables))

    return {'c_loss' : c_loss, 'g_loss' : g_loss}


def critic_loss(real_logits, fake_logits):
    real_loss = tf.reduce_mean(real_logits)
    fake_loss = tf.reduce_mean(fake_logits)
    return (fake_loss - real_loss)

def generator_loss(fake_logits):
    return -tf.reduce_mean(fake_logits)

generator_optimizer = keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.5, beta_2=0.9
)
critic_optimizer = keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.5, beta_2=0.9
)
