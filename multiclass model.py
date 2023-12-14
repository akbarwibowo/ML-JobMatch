
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# X1 = Job Experience Required
# X2 = undergrad_degree
# X3 = Key Skills
# labels = Job Title

column_job_experience = []
column_undergrad = []
column_key_skills = []
column_job_title = []

X1 = np.random.choice(10, 426)  # 10 pilihan (minimal 1 tahun, maksimal 10 tahun), total 426 data, numerik
X2 = np.random.choice(column_undergrad, 426)  # kategori
X3 = np.random.choice(column_key_skills, 426)  # kategori
labels = np.random.choice(column_job_title, 426)

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

cat_encoder = OneHotEncoder(sparse=False)
X2_encoded = cat_encoder.fit_transform(X2.reshape(-1, 1))
X3_encoded = cat_encoder.fit_transform(X3.reshape(-1, 1))

X1_train, X1_test, X2_train, X2_test, X3_train, X3_test, y_train, y_test = train_test_split(
    X1, X2_encoded, X3_encoded, encoded_labels, test_size=0.4, random_state=35
)

# Input layers
input1 = tf.keras.layers.Input(shape=(X1.shape[1],), name='input1')  # X1, numeric
input2 = tf.keras.layers.Input(shape=(X2_encoded.shape[1],), name='input2')  # X2, categorical
input3 = tf.keras.layers.Input(shape=(X3_encoded.shape[1],), name='input3')  # X3, categorical

concatenated = tf.keras.layers.concatenate([input1, input2, input3])

merged = tf.keras.layers.Dense(128, activation='relu')(concatenated)
output_layer = tf.keras.layers.Dense(len(np.unique(encoded_labels)), activation='softmax', name='output')(merged)

# Model functional API
model = tf.keras.models.Model(inputs=[input1, input2, input3], outputs=output_layer)


learning_rate = 0.0035
batch_size = 32

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit([X1_train, X2_train, X3_train], y_train, epochs=30, batch_size=batch_size, validation_split=0.2)

# Evaluate the model
test_loss, test_accuracy = model.evaluate([X1_test, X2_test, X3_test], y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')