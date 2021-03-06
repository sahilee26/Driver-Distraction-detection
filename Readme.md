
# Distracted Driver Detection

### Problem Definition:
Given Statefarm dataset of 2D dashboard car images , this kaggle challenge gives task of classifying each driver's behavior.


### Examples:
<img src="examples/img1.jpg" width="200"> <img src="examples/img2.jpg" width="200"> <img src="examples/img3.jpg" width="200">

The images are labeled following a set of 10 categories:
|Class|Description|
|-----|-----------|
| `c0` | Safe driving. |
| `c1` | Texting (right hand). |
| `c2` | Talking on the phone (right hand). |
| `c3` | Texting (left hand). |
| `c4` | Talking on the phone (left hand). |
| `c5` | Operating the radio. |
| `c6` | Drinking. |
| `c7` | Reaching behind. |
| `c8` | Hair and makeup. |
| `c9` | Talking to passenger(s). |

# Brief 
I used keras for importing VGG model and then added extra layers(4 dense, 2 dropout and Average Pooling) on top of network that I trained using the dataset.

### Dependencies

* `Python 3.6.1`
* `Tensorflow 1.3.0`
* `Keras 2.1.2`
* `matplotlib 2.0.2`
* `numpy 1.12.1`
