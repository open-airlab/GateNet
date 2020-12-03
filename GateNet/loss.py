def loss(y_true, y_pred):
    
    p_pred = y_pred[:,:,:,0]
    cx_pred = y_pred[:,:,:,1]
    cy_pred = y_pred[:,:,:,2]
    d_pred = y_pred[:,:,:,3]
    o_pred = y_pred[:,:,:,4]


    p_true = y_true[:,:,:,0]
    cx_true = y_true[:,:,:,1]
    cy_true = y_true[:,:,:,2]
    d_true = y_true[:,:,:,3]
    o_true = y_true[:,:,:,4]

    xy_loss = tf.math.multiply(p_true, tf.math.square(cx_true - cx_pred)+tf.math.square(cy_true - cy_pred))

    d_loss = tf.math.multiply(p_true, tf.math.square((d_true) - (d_pred)))
    o_loss = tf.math.multiply(p_true, tf.math.square((o_true) - (o_pred)))


    p_loss = tf.math.multiply(p_true, tf.math.square(p_pred - p_true))
    nop_loss = tf.math.multiply(1-p_true, tf.math.square(p_pred - p_true))

    xy_loss = tf.keras.backend.sum(xy_loss, axis=-1)
    p_loss = tf.keras.backend.sum(p_loss, axis=-1)
    nop_loss = tf.keras.backend.sum(nop_loss, axis=-1)
    d_loss = tf.keras.backend.sum(d_loss, axis=-1)
    o_loss = tf.keras.backend.sum(o_loss, axis=-1)

    total_loss = 5*xy_loss  + 5*p_loss + 0.5*nop_loss+ 5*d_loss + 5*o_loss

    return total_loss
