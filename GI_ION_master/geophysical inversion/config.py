'''Global configuration for the experiments'''

######################
#  General settings  #
######################

# Filename to save the model under
filename_out    = './models/output/my_inn.pt'
# Model to load and continue training. Ignored if empty string
filename_in     = ''
# Compute device to perform the training on, 'cuda' or 'cpu'
device          = 'cuda:0'
# Use interactive visualization of losses and other plots. Requires visdom
interactive_visualization = True


#######################
#  Training schedule  #
#######################

# Initial learning rate
lr_init         = 1.0e-5
#Batch size
batch_size      = 500
# Total number of epochs to train for
n_epochs        = 60
# End the epoch after this many iterations (or when the train loader is exhausted)
n_its_per_epoch = 200
# For the first n epochs, train with a much lower learning rate. This can be
# helpful if the model immediately explodes.
pre_low_lr      = 0
# Decay exponentially each epoch, to final_decay*lr_init at the last epoch.
final_decay     = 0.02
# L2 weight regularization of model parameters
l2_weight_reg   = 1e-5
# Parameters beta1, beta2 of the Adam optimizer
adam_betas = (0.9, 0.95)

#####################
#  Data dimensions  #
#####################

ndim_x     = 10
ndim_pad_x = 0

ndim_y     = 5
ndim_z     = 5
ndim_pad_zy = 0

train_loader, test_loader = None, None

############
#  Losses  #
############

interactive_visualization = False
show_test_loss = True
show_total_loss = False

train_max_likelihood = True
train_forward_fit       = True
train_independence_loss = True


lambd_max_likelihood = 0.1
lambd_fit_forw       = 10
lambd_independence_loss    = 0.1

# noise of this sigma
add_y_noise     = 5e-2
# In all cases, perturb the zero padding
add_pad_noise   = 1e-2



###########
#  Model  #
###########

# Initialize the model parameters from a normal distribution with this sigma
init_scale = 0.001
#
N_blocks   = 4
#
exponent_clamping = 4.0
#
hidden_layer_sizes = 128
#
use_permutation = True
#
verbose_construction = False


