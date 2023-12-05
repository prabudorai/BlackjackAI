import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load data
totals = open("data\\totals.txt").readlines()
tags = open("data\\actions_log.txt").readlines()

data_clean = []
tags_clean = []

for datum in totals:
	clean_datum = datum[:datum.index('\n')].strip()
	clean_datum = clean_datum[1:-1].split(',')
	clean_datum[0] = int( clean_datum[0] )
	clean_datum[1] = int( clean_datum[1] )
	clean_datum[2] = int( clean_datum[2] )
	data_clean = data_clean + [ clean_datum ]

for tag in tags:
	tag = tag[:tag.index('\n')]
	if tag == "h":
		tags_clean = tags_clean + [ 1 ]
	else:
		tags_clean = tags_clean + [ 0 ]


size =  len(totals)


train_data = np.array( data_clean[:int(size*.8)] )
train_tags = np.array( tags_clean[:int(size*.8)] )

val_data = np.array( data_clean[int(size*.8):int(size*.9)] )
val_tags = np.array( tags_clean[int(size*.8):int(size*.9)] )

test_data = np.array( data_clean[int(size*.9):] )
test_tags = np.array( tags_clean[int(size*.9):] )

model = keras.Sequential()
model.add( keras.layers.Dense( 3, input_dim=3 ) )
model.add( keras.layers.Dense( 16, activation = 'relu' ) )
model.add( keras.layers.Dense( 2, activation='softmax') )

model.compile(optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

history = model.fit(train_data, train_tags, epochs=100, validation_data=(val_data,val_tags))

train_loss = history.history['loss']
train_acc = history.history['accuracy']

validation_loss = history.history['val_loss']
validation_acc = history.history['val_accuracy']

test_loss, test_acc = model.evaluate(test_data, test_tags)

print('Test accuracy:', test_acc)


plt.figure(figsize= (10,5))

plt.subplot(2,2,1)
plt.plot(train_loss, label="Train Loss")
plt.plot(validation_loss, label="Validation Loss")

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2,2,2)
plt.plot(train_acc, label="Train Accuracy")
plt.plot(validation_acc, label="Validation Accuracy")

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()



# Print Hard hand table
results = []

for k,i in enumerate(range(4,22)):
	results = results + [ "" ]
	for j in range(1,11):
		prediction = model.predict( np.array([ [i,j, 0] ] ) )
		if prediction[0][0] > prediction[0][1]:
			results[k] = results[k] + "s"
		else:
			results[k] = results[k] + "h"
            
print('Hard Hands')
print('---------------------------')
print( "  ", end="" )
for x in range( len(results[0]) ):
	print( " " + str( (x+2)%10 ), end="" )
print( )
for i in range( len(results) ):
	print( i+4, end="" )
	if i+4 < 10:
		print( "  ", end="" )
	else:
		print( " ", end="" )
	for j in range( len(results[i] ) ):
		print( results[i][j], end=" " )

	print( )
print('---------------------------') 

 
# Print Soft hand table
results = []

for k,i in enumerate(range(2,11)):
    results = results + [ "" ]
    for j in range(1,11):
        prediction = model.predict( np.array([ [i+11,j, 1] ] ) )
        if prediction[0][0] > prediction[0][1]:
            results[k] = results[k] + "s"
        else:
            results[k] = results[k] + "h"
print('Soft Hands')
print('---------------------------')
print( "  ", end="" )
for x in range( len(results[0]) ):
    print( " " + str( (x+2)%10 ), end="" )
print( )
for i in range( len(results) ):
    print( i+2, end="" )
    if i+2 < 10:
        print( "  ", end="" )
    else:
        print( " ", end="" )
    for j in range( len(results[i] ) ):
        print( results[i][j], end=" " )
    print( )


plt.tight_layout()
plt.show()