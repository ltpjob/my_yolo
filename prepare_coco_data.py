from pycocotools.coco import COCO
import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np

dataDir = 'F:/image_data/coco/annotations_trainval2017'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)
coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))
nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['person'])
print(catIds)
imgIds = coco.getImgIds(catIds=catIds )
imgIds = coco.getImgIds(imgIds =imgIds)
print(imgIds, len(imgIds), np.random.randint(0,len(imgIds)))
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]
I = io.imread(img['coco_url'])
plt.axis('off')
plt.imshow(I)
plt.show()