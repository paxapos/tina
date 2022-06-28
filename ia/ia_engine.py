import os 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class IaEngine:
    def train(self):
        print('Entrenando')
        pass
    def predict(self):
        pass


engine = IaEngine()
#engine.train()



base_dir = 'Insert "pics" path' 
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')


train_raw_dir = os.path.join(train_dir, 'raw')

train_burned_dir = os.path.join(train_dir, 'burned')

validation_raw_dir = os.path.join(validation_dir, 'raw')

validation_burned_dir = os.path.join(validation_dir, 'burned')


train_raw_fnames = os.listdir(train_raw_dir)

train_burned_fnames = os.listdir(train_burned_dir)


# Parameters for the matplotlib graph
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Set up matplotlib fig and size it to fit 4x4 pics
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_raw_pix = [os.path.join(train_raw_dir, fname) 
                for fname in train_raw_fnames[pic_index-8:pic_index]]
next_burned_pix = [os.path.join(train_burned_dir, fname) 
                for fname in train_burned_fnames[pic_index-8:pic_index]]

for i, img_path in enumerate(next_raw_pix+next_burned_pix):
  # Set up subplot
  sp = plt.subplot(nrows, ncols, i + 1)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
