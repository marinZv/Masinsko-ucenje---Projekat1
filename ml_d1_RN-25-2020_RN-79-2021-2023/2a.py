# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

# Izbegavamo scientific notaciju i zaokruzujemo na 5 decimala.
# np.set_printoptions(suppress=True, precision=5)

# Učitavanje i obrada podataka.
filename = 'funky.csv'
all_data = np.loadtxt(filename, delimiter=',', usecols=(0, 1), dtype='float32')
data = dict()
data['x'] = all_data[:, 0]
data['y'] = all_data[:, 1]
costs = []

# Nasumično mešanje.
nb_samples = data['x'].shape[0]
indices = np.random.permutation(nb_samples)
data['x'] = data['x'][indices]
data['y'] = data['y'][indices]

# Normalizacija (obratiti pažnju na axis=0).
data['x'] = (data['x'] - np.mean(data['x'], axis=0)) / np.std(data['x'], axis=0)
data['y'] = (data['y'] - np.mean(data['y'])) / np.std(data['y'])

# Kreiranje feature matrice.
# Ovom promenljivom kontrolisemo broj feature-a tj. stepen polinoma. Varirati!
# nb_features = 1, avg loss u 1000toj epohi = 0.42
# nb_features = 9, avg loss u 1000toj epohi = 0.33
# Overfitting, regularizacija...
nb_features = 1
print('Originalne vrednosti (prve 3):')
print(data['x'][:3])
print('Feature matrica (prva 3 reda):')
data['x'] = create_feature_matrix(data['x'], nb_features)
print(data['x'][:3, :])

# Iscrtavanje.
plt.scatter(data['x'][:, 0], data['y'])


# Model i parametri.
w = tf.Variable(tf.zeros(nb_features))
b = tf.Variable(0.0)

learning_rate = 0.001
nb_epochs = 1000

def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    return hyp

# # Funkcija troška i optimizacija.
def loss(x, y, w, b, reg=None):
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    mse = tf.reduce_mean(tf.square(prediction - y_col))

    loss = mse

    return loss

# # Računanje gradijenta
def calc_grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, reg='l2')

    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val

# # Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa
# # slozenijim funkcijama.
adam = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# # Trening korak
def train_step(x, y, w, b):
    w_grad, b_grad, loss = calc_grad(x, y, w, b)

    adam.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss

# Trening.
for epoch in range(nb_epochs):

    # Stochastic Gradient Descent.
    epoch_loss = 0
    for sample in range(nb_samples):
        x = data['x'][sample].reshape((1, nb_features))
        y = data['y'][sample]

        curr_loss = train_step(x, y, w, b)
        epoch_loss += curr_loss

    # U svakoj stotoj epohi ispisujemo prosečan loss.
    epoch_loss /= nb_samples
    if (epoch + 1) % 100 == 0:
      print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

curr_loss=0
for sample in range(nb_samples):
    x = data['x'][sample].reshape((1, nb_features))
    y = data['y'][sample]
    curr_loss+=train_step(x,y,w,b)

costs.append(curr_loss)

# Ispisujemo i plotujemo finalnu vrednost parametara.
print(f'w = {w.numpy()}, bias = {b.numpy()}')
xs = create_feature_matrix(np.linspace(-2, 4, 100, dtype='float32'),
nb_features)
hyp_val = pred(xs, w, b)
plt.plot(xs[:, 0].tolist(), hyp_val.numpy().tolist(), color='g')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

# #######################                 #################
# #######################         2       #################
# #######################                 #################

# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix2(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

# Izbegavamo scientific notaciju i zaokruzujemo na 5 decimala.
# np.set_printoptions(suppress=True, precision=5)

# Učitavanje i obrada podataka.
filename2 = 'funky.csv'
all_data2 = np.loadtxt(filename, delimiter=',', usecols=(0, 1), dtype='float32')
data2 = dict()
data2['x'] = all_data2[:, 0]
data2['y'] = all_data2[:, 1]

# Nasumično mešanje.
nb_samples2 = data2['x'].shape[0]
indices2 = np.random.permutation(nb_samples2)
data2['x'] = data2['x'][indices2]
data2['y'] = data2['y'][indices2]

# Normalizacija (obratiti pažnju na axis=0).
data2['x'] = (data2['x'] - np.mean(data2['x'], axis=0)) /
np.std(data2['x'], axis=0)
data2['y'] = (data2['y'] - np.mean(data2['y'])) / np.std(data2['y'])

# Kreiranje feature matrice.
# Ovom promenljivom kontrolisemo broj feature-a tj. stepen polinoma. Varirati!
# nb_features = 1, avg loss u 1000toj epohi = 0.42
# nb_features = 9, avg loss u 1000toj epohi = 0.33
# Overfitting, regularizacija...
nb_features2 = 2
print('Originalne vrednosti (prve 3):')
print(data2['x'][:3])
print('Feature matrica (prva 3 reda):')
data2['x'] = create_feature_matrix2(data2['x'], nb_features2)
print(data2['x'][:3, :])

# Iscrtavanje.
plt.scatter(data2['x'][:, 0], data2['y'])


# Model i parametri.
w2 = tf.Variable(tf.zeros(nb_features2))
b2 = tf.Variable(0.0)

learning_rate2 = 0.001
nb_epochs2 = 1000

def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features2, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    return hyp

# # Funkcija troška i optimizacija.
def loss(x, y, w, b, reg=None):
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    mse = tf.reduce_mean(tf.square(prediction - y_col))

    loss = mse

    return loss

# # Računanje gradijenta
def calc_grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, reg='l2')

    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val

# # Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa
# # slozenijim funkcijama.
adam2 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# # Trening korak
def train_step(x, y, w, b):
    w_grad, b_grad, loss = calc_grad(x, y, w, b)

    adam2.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss

# Trening.
for epoch in range(nb_epochs2):

    # Stochastic Gradient Descent.
    epoch_loss = 0
    for sample in range(nb_samples2):
        x2 = data2['x'][sample].reshape((1, nb_features2))
        y2 = data2['y'][sample]

        curr_loss = train_step(x2, y2, w2, b2)
        epoch_loss += curr_loss

    # U svakoj stotoj epohi ispisujemo prosečan loss.
    epoch_loss /= nb_samples2
    if (epoch + 1) % 100 == 0:
      print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

curr_loss=0
for sample in range(nb_samples):
    x = data2['x'][sample].reshape((1, nb_features2))
    y = data2['y'][sample]
    curr_loss+=train_step(x,y,w2,b2)

costs.append(curr_loss)

# Ispisujemo i plotujemo finalnu vrednost parametara.
print(f'w = {w2.numpy()}, bias = {b2.numpy()}')
xs2 = create_feature_matrix2(np.linspace(-2, 4, 100, dtype='float32'),
nb_features2)
hyp_val2 = pred(xs2, w2, b2)
plt.plot(xs2[:, 0].tolist(), hyp_val2.numpy().tolist(), color='r')
plt.xlim([-3, 3])
plt.ylim([-3, 3])


#######################                 #################
#######################         3       #################
#######################                 #################

# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix3(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

# Izbegavamo scientific notaciju i zaokruzujemo na 5 decimala.
# np.set_printoptions(suppress=True, precision=5)

# Učitavanje i obrada podataka.
filename3 = 'funky.csv'
all_data3 = np.loadtxt(filename3, delimiter=',', usecols=(0, 1),
dtype='float32')
data3 = dict()
data3['x'] = all_data3[:, 0]
data3['y'] = all_data3[:, 1]

# Nasumično mešanje.
nb_samples3 = data3['x'].shape[0]
indices3 = np.random.permutation(nb_samples3)
data3['x'] = data3['x'][indices3]
data3['y'] = data3['y'][indices3]

# Normalizacija (obratiti pažnju na axis=0).
data3['x'] = (data3['x'] - np.mean(data3['x'], axis=0)) /
np.std(data3['x'], axis=0)
data3['y'] = (data3['y'] - np.mean(data3['y'])) / np.std(data3['y'])

# Kreiranje feature matrice.
# Ovom promenljivom kontrolisemo broj feature-a tj. stepen polinoma. Varirati!
# nb_features = 1, avg loss u 1000toj epohi = 0.42
# nb_features = 9, avg loss u 1000toj epohi = 0.33
# Overfitting, regularizacija...
nb_features3 = 3
print('Originalne vrednosti (prve 3):')
print(data3['x'][:3])
print('Feature matrica (prva 3 reda):')
data3['x'] = create_feature_matrix3(data3['x'], nb_features3)
print(data3['x'][:3, :])

# Iscrtavanje.
plt.scatter(data3['x'][:, 0], data3['y'])


# Model i parametri.
w3 = tf.Variable(tf.zeros(nb_features3))
b3 = tf.Variable(0.0)

learning_rate3 = 0.001
nb_epochs3 = 1000

def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features3, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    return hyp

# # Funkcija troška i optimizacija.
def loss(x, y, w, b, reg=None):
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    mse = tf.reduce_mean(tf.square(prediction - y_col))

    loss = mse

    return loss

# # Računanje gradijenta
def calc_grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, reg='l2')

    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val

# # Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa
# # slozenijim funkcijama.
adam3 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# # Trening korak
def train_step(x, y, w, b):
    w_grad, b_grad, loss = calc_grad(x, y, w, b)

    adam3.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss

# Trening.
for epoch in range(nb_epochs3):

    # Stochastic Gradient Descent.
    epoch_loss = 0
    for sample in range(nb_samples3):
        x3 = data3['x'][sample].reshape((1, nb_features3))
        y3 = data3['y'][sample]

        curr_loss = train_step(x3, y3, w3, b3)
        epoch_loss += curr_loss

    # U svakoj stotoj epohi ispisujemo prosečan loss.
    epoch_loss /= nb_samples3
    if (epoch + 1) % 100 == 0:
      print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

curr_loss=0
for sample in range(nb_samples):
    x = data3['x'][sample].reshape((1, nb_features3))
    y = data3['y'][sample]
    curr_loss+=train_step(x,y,w3,b3)

costs.append(curr_loss)

# Ispisujemo i plotujemo finalnu vrednost parametara.
print(f'w = {w3.numpy()}, bias = {b3.numpy()}')
xs3 = create_feature_matrix3(np.linspace(-2, 4, 100, dtype='float32'),
nb_features3)
hyp_val3 = pred(xs3, w3, b3)
plt.plot(xs3[:, 0].tolist(), hyp_val3.numpy().tolist(), color='b')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

#######################                 #################
#######################         4       #################
#######################                 #################

# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix4(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

# Izbegavamo scientific notaciju i zaokruzujemo na 5 decimala.
# np.set_printoptions(suppress=True, precision=5)

# Učitavanje i obrada podataka.
filename4 = 'funky.csv'
all_data4 = np.loadtxt(filename4, delimiter=',', usecols=(0, 1),
dtype='float32')
data4 = dict()
data4['x'] = all_data4[:, 0]
data4['y'] = all_data4[:, 1]

# Nasumično mešanje.
nb_samples4 = data4['x'].shape[0]
indices4 = np.random.permutation(nb_samples4)
data4['x'] = data4['x'][indices4]
data4['y'] = data4['y'][indices4]

# Normalizacija (obratiti pažnju na axis=0).
data4['x'] = (data4['x'] - np.mean(data4['x'], axis=0)) /
np.std(data4['x'], axis=0)
data4['y'] = (data4['y'] - np.mean(data4['y'])) / np.std(data4['y'])

# Kreiranje feature matrice.
# Ovom promenljivom kontrolisemo broj feature-a tj. stepen polinoma. Varirati!
# nb_features = 1, avg loss u 1000toj epohi = 0.42
# nb_features = 9, avg loss u 1000toj epohi = 0.33
# Overfitting, regularizacija...
nb_features4 = 4
print('Originalne vrednosti (prve 3):')
print(data4['x'][:3])
print('Feature matrica (prva 3 reda):')
data4['x'] = create_feature_matrix4(data4['x'], nb_features4)
print(data4['x'][:3, :])

# Iscrtavanje.
plt.scatter(data4['x'][:, 0], data4['y'])

# Model i parametri.
w4 = tf.Variable(tf.zeros(nb_features4))
b4 = tf.Variable(0.0)

learning_rate4 = 0.001
nb_epochs4 = 1000

def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features4, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    return hyp

# # Funkcija troška i optimizacija.
def loss(x, y, w, b, reg=None):
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    mse = tf.reduce_mean(tf.square(prediction - y_col))

    loss = mse

    return loss

# # Računanje gradijenta
def calc_grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, reg='l2')

    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val

# # Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa
# # slozenijim funkcijama.
adam4 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# # Trening korak
def train_step(x, y, w, b):
    w_grad, b_grad, loss = calc_grad(x, y, w, b)

    adam4.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss

# Trening.
for epoch in range(nb_epochs4):

    # Stochastic Gradient Descent.
    epoch_loss = 0
    for sample in range(nb_samples4):
        x4 = data4['x'][sample].reshape((1, nb_features4))
        y4 = data4['y'][sample]

        curr_loss = train_step(x4, y4, w4, b4)
        epoch_loss += curr_loss

    # U svakoj stotoj epohi ispisujemo prosečan loss.
    epoch_loss /= nb_samples4
    if (epoch + 1) % 100 == 0:
      print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

curr_loss=0
for sample in range(nb_samples):
    x = data4['x'][sample].reshape((1, nb_features4))
    y = data4['y'][sample]
    curr_loss+=train_step(x,y,w4,b4)

costs.append(curr_loss)

# Ispisujemo i plotujemo finalnu vrednost parametara.
print(f'w = {w4.numpy()}, bias = {b4.numpy()}')
xs4 = create_feature_matrix4(np.linspace(-2, 4, 100, dtype='float32'),
nb_features4)
hyp_val4 = pred(xs4, w4, b4)
plt.plot(xs4[:, 0].tolist(), hyp_val4.numpy().tolist(), color='y')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

#######################                 #################
#######################         5       #################
#######################                 #################

# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix5(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

# Izbegavamo scientific notaciju i zaokruzujemo na 5 decimala.
# np.set_printoptions(suppress=True, precision=5)

# Učitavanje i obrada podataka.
filename5 = 'funky.csv'
all_data5 = np.loadtxt(filename5, delimiter=',', usecols=(0, 1),
dtype='float32')
data5 = dict()
data5['x'] = all_data5[:, 0]
data5['y'] = all_data5[:, 1]

# Nasumično mešanje.
nb_samples5 = data5['x'].shape[0]
indices5 = np.random.permutation(nb_samples5)
data5['x'] = data5['x'][indices5]
data5['y'] = data5['y'][indices5]

# Normalizacija (obratiti pažnju na axis=0).
data5['x'] = (data5['x'] - np.mean(data5['x'], axis=0)) /
np.std(data5['x'], axis=0)
data5['y'] = (data5['y'] - np.mean(data5['y'])) / np.std(data5['y'])

# Kreiranje feature matrice.
# Ovom promenljivom kontrolisemo broj feature-a tj. stepen polinoma. Varirati!
# nb_features = 1, avg loss u 1000toj epohi = 0.42
# nb_features = 9, avg loss u 1000toj epohi = 0.33
# Overfitting, regularizacija...
nb_features5 = 5
print('Originalne vrednosti (prve 3):')
print(data5['x'][:3])
print('Feature matrica (prva 3 reda):')
data5['x'] = create_feature_matrix5(data5['x'], nb_features5)
print(data5['x'][:3, :])

# Iscrtavanje.
plt.scatter(data5['x'][:, 0], data5['y'])


# Model i parametri.
w5 = tf.Variable(tf.zeros(nb_features5))
b5 = tf.Variable(0.0)

learning_rate5 = 0.001
nb_epochs5 = 1000

def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features5, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    return hyp

# # Funkcija troška i optimizacija.
def loss(x, y, w, b, reg=None):
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    mse = tf.reduce_mean(tf.square(prediction - y_col))

    loss = mse

    return loss

# # Računanje gradijenta
def calc_grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, reg='l2')

    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val

# # Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa
# # slozenijim funkcijama.
adam5 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# # Trening korak
def train_step(x, y, w, b):
    w_grad, b_grad, loss = calc_grad(x, y, w, b)

    adam5.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss

# Trening.
for epoch in range(nb_epochs5):

    # Stochastic Gradient Descent.
    epoch_loss = 0
    for sample in range(nb_samples5):
        x5 = data5['x'][sample].reshape((1, nb_features5))
        y5 = data5['y'][sample]

        curr_loss = train_step(x5, y5, w5, b5)
        epoch_loss += curr_loss

    # U svakoj stotoj epohi ispisujemo prosečan loss.
    epoch_loss /= nb_samples5
    if (epoch + 1) % 100 == 0:
      print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

curr_loss=0
for sample in range(nb_samples):
    x = data5['x'][sample].reshape((1, nb_features5))
    y = data5['y'][sample]
    curr_loss+=train_step(x,y,w5,b5)

costs.append(curr_loss)

# Ispisujemo i plotujemo finalnu vrednost parametara.
print(f'w = {w5.numpy()}, bias = {b5.numpy()}')
xs5 = create_feature_matrix5(np.linspace(-2, 4, 100, dtype='float32'),
nb_features5)
hyp_val5 = pred(xs5, w5, b5)
plt.plot(xs5[:, 0].tolist(), hyp_val5.numpy().tolist(), color='m')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

#######################                 #################
#######################         6       #################
#######################                 #################

# Pomocna funkcija koja od niza trening primera pravi feature matricu (m X n).
def create_feature_matrix6(x, nb_features):
  tmp_features = []
  for deg in range(1, nb_features+1):
    tmp_features.append(np.power(x, deg))
  return np.column_stack(tmp_features)

# Izbegavamo scientific notaciju i zaokruzujemo na 5 decimala.
# np.set_printoptions(suppress=True, precision=5)

# Učitavanje i obrada podataka.
filename6 = 'funky.csv'
all_data6 = np.loadtxt(filename6, delimiter=',', usecols=(0, 1),
dtype='float32')
data6 = dict()
data6['x'] = all_data6[:, 0]
data6['y'] = all_data6[:, 1]

# Nasumično mešanje.
nb_samples6 = data6['x'].shape[0]
indices6 = np.random.permutation(nb_samples6)
data6['x'] = data6['x'][indices6]
data6['y'] = data6['y'][indices6]

# Normalizacija (obratiti pažnju na axis=0).
data6['x'] = (data6['x'] - np.mean(data6['x'], axis=0)) /
np.std(data6['x'], axis=0)
data6['y'] = (data6['y'] - np.mean(data6['y'])) / np.std(data6['y'])

# Kreiranje feature matrice.
# Ovom promenljivom kontrolisemo broj feature-a tj. stepen polinoma. Varirati!
# nb_features = 1, avg loss u 1000toj epohi = 0.42
# nb_features = 9, avg loss u 1000toj epohi = 0.33
# Overfitting, regularizacija...
nb_features6 = 6
print('Originalne vrednosti (prve 3):')
print(data6['x'][:3])
print('Feature matrica (prva 3 reda):')
data6['x'] = create_feature_matrix6(data6['x'], nb_features6)
print(data6['x'][:3, :])

# Iscrtavanje.
plt.scatter(data6['x'][:, 0], data6['y'])

# Model i parametri.
w6 = tf.Variable(tf.zeros(nb_features6))
b6 = tf.Variable(0.0)

learning_rate6 = 0.001
nb_epochs6 = 1000

def pred(x, w, b):
    w_col = tf.reshape(w, (nb_features6, 1))
    hyp = tf.add(tf.matmul(x, w_col), b)
    return hyp

# # Funkcija troška i optimizacija.
def loss(x, y, w, b, reg=None):
    prediction = pred(x, w, b)

    y_col = tf.reshape(y, (-1, 1))
    mse = tf.reduce_mean(tf.square(prediction - y_col))

    loss = mse

    return loss

# # Računanje gradijenta
def calc_grad(x, y, w, b):
    with tf.GradientTape() as tape:
        loss_val = loss(x, y, w, b, reg='l2')

    w_grad, b_grad = tape.gradient(loss_val, [w, b])

    return w_grad, b_grad, loss_val

# # Prelazimo na AdamOptimizer jer se prost GradientDescent lose snalazi sa
# # slozenijim funkcijama.
adam6 = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# # Trening korak
def train_step(x, y, w, b):
    w_grad, b_grad, loss = calc_grad(x, y, w, b)

    adam6.apply_gradients(zip([w_grad, b_grad], [w, b]))

    return loss

# Trening.
for epoch in range(nb_epochs6):

    # Stochastic Gradient Descent.
    epoch_loss = 0
    for sample in range(nb_samples6):
        x6 = data6['x'][sample].reshape((1, nb_features6))
        y6 = data6['y'][sample]

        curr_loss = train_step(x6, y6, w6, b6)
        epoch_loss += curr_loss

    # U svakoj stotoj epohi ispisujemo prosečan loss.
    epoch_loss /= nb_samples6
    if (epoch + 1) % 100 == 0:
      print(f'Epoch: {epoch+1}/{nb_epochs}| Avg loss: {epoch_loss:.5f}')

curr_loss=0
for sample in range(nb_samples):
    x = data6['x'][sample].reshape((1, nb_features6))
    y = data6['y'][sample]
    curr_loss+=train_step(x,y,w6,b6)

costs.append(curr_loss)

# Ispisujemo i plotujemo finalnu vrednost parametara.
print(f'w = {w6.numpy()}, bias = {b6.numpy()}')
xs6 = create_feature_matrix6(np.linspace(-2, 4, 100, dtype='float32'),
nb_features6)
hyp_val6 = pred(xs6, w6, b6)
plt.plot(xs6[:, 0].tolist(), hyp_val6.numpy().tolist(), color='c')
plt.xlim([-3, 3])
plt.ylim([-3, 3])

plt.show()

plt.plot(range(1,7),costs)
plt.show()

