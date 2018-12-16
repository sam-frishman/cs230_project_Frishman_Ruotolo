"""Define the model."""

import tensorflow as tf


def build_model(is_training, inputs, params):
    """Compute logits of the model (output distribution)

    Args:
        is_training: (bool) whether we are training or not
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) hyperparameters

    Returns:
        output: (tf.Tensor) output of the model
    """
    images = inputs['images']

    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 3]

    out = images
    # Define the number of channels of each convolution
    # For each block, we do: 3x3 conv -> batch norm -> relu -> 2x2 maxpool
   
    
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum

    """
    channels = [num_channels, num_channels * 2, num_channels * 4, num_channels * 8, num_channels * 16]
    for i, c in enumerate(channels):
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, c, 3, padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)
    
    finalDims = 4
    fNum5 = num_channels*16
    #conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True)
    """
    imageSize = 128
    numFilters = 5
    finalDims = int(imageSize/2**(numFilters))
    fNum1 = 16
    fNum2 = 16*2
    fNum3 = 16*4
    fNum4 = 16*8
    fNum5 = 16*16
    fNum6 = 16*16
    k1, k2, k3, k4, k5, k6 = 2,2,2,2,2,2
    with tf.variable_scope('block_{}'.format(1)):
	# Filter 1 
        out = tf.layers.conv2d(out, fNum1, k1, padding='same')
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)

	# Filter 2 
        with tf.variable_scope('block_{}'.format(1)):
        out = tf.layers.conv2d(out, fNum2, k2, padding='same')
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)
	
	# Filter 3 
        out = tf.layers.conv2d(out, fNum3, k3, padding='same')
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        #out = tf.layers.max_pooling2d(out, 2, 2)

	# Filter 4 
        out = tf.layers.conv2d(out, fNum4, k4, padding='same')
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)
	
	# Filter 5 
        out = tf.layers.conv2d(out, fNum5, k5, padding='same')
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)

        # Filter 6
        out = tf.layers.conv2d(out, fNum6, k6, padding='same')
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)

    numFilters = 6
    imageSize = 128
    kernels = [2, 2, 2, 2, 2, 2]
    channels = [16, 16*2, 16*4, 16*8, 16*16, 16*32]
    maxPool = [True, True, False, True, True, True]
    numPools = 5
    finalDims = imageSize/(numPools*2)
    for i in range(numFilters)
        with tf.variable_scope('block_{}'.format(i+1)):
            out = tf.layers.conv2d(out, channels[i], kernels[i], padding='same')
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
            out = tf.nn.relu(out)
	    if maxPool[i]:
		out = tf.layers.max_pooling2d(out, 2, 2)


    """
    num_channels = params.num_channels
    bn_momentum = params.bn_momentum

    #conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True)
    # layer 1, i = 0?, channels = 1, filter dim = 4
    with tf.variable_scope('block_{}'.format(1)):
        out = tf.layers.conv2d(out, filters=4, kernel_size=2, strides=2, padding='same')
        out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
        out = tf.layers.max_pooling2d(out, 2, 2)

    """

    print(out.get_shape().as_list())
    print(finalDims)

    assert out.get_shape().as_list() == [None, finalDims, finalDims, channels[numFilters-1]]

    out = tf.reshape(out, [-1, finalDims*finalDims*fNum5])
    with tf.variable_scope('fc_1'):
        out = tf.layers.dense(out, num_channels * 16)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        out = tf.nn.relu(out)
    with tf.variable_scope('fc_2'):
        out = tf.layers.dense(out, num_channels * 8)
        if params.use_batch_norm:
            out = tf.layers.batch_normalization(out, momentum=bn_momentum, training=is_training)
        logits = tf.nn.relu(out)

    return logits


def model_fn(mode, inputs, params, reuse=False):
    """Model function defining the graph operations.

    Args:
        mode: (string) can be 'train' or 'eval'
        inputs: (dict) contains the inputs of the graph (features, labels...)
                this can be `tf.placeholder` or outputs of `tf.data`
        params: (Params) contains hyperparameters of the model (ex: `params.learning_rate`)
        reuse: (bool) whether to reuse the weights

    Returns:
        model_spec: (dict) contains the graph operations or nodes needed for training / evaluation
    """
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    # -----------------------------------------------------------
    # MODEL: define the layers of the model
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        logits = build_model(is_training, inputs, params)
        predictions = tf.argmax(logits, 1)

    # Define loss and accuracy
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(labels, predictions), tf.float32))

    # Define training step that minimizes the loss with the Adam optimizer
    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            # Add a dependency to update the moving mean and variance for batch normalization
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)


    # -----------------------------------------------------------
    # METRICS AND SUMMARIES
    # Metrics for evaluation using tf.metrics (average over whole dataset)
    with tf.variable_scope("metrics"):
        metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=tf.argmax(logits, 1)),
            'loss': tf.metrics.mean(loss)
        }

    # Group the update ops for the tf.metrics
    update_metrics_op = tf.group(*[op for _, op in metrics.values()])

    # Get the op to reset the local variables used in tf.metrics
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    # Summaries for training
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    tf.summary.image('train_image', inputs['images'])

    #TODO: if mode == 'eval': ?
    # Add incorrectly labeled images
    mask = tf.not_equal(labels, predictions)

    # Add a different summary to know how they were misclassified
    for label in range(0, params.num_labels):
        mask_label = tf.logical_and(mask, tf.equal(predictions, label))
        incorrect_image_label = tf.boolean_mask(inputs['images'], mask_label)
        tf.summary.image('incorrectly_labeled_{}'.format(label), incorrect_image_label)

    # -----------------------------------------------------------
    # MODEL SPECIFICATION
    # Create the model specification and return it
    # It contains nodes or operations in the graph that will be used for training and evaluation
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec["predictions"] = predictions
    model_spec['loss'] = loss
    model_spec['accuracy'] = accuracy
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
