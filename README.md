The raw data is quite large (approximately 413 GB before compression). As a result, the raw data is not publicly available at this time but may be obtained from the authors upon reasonable request.

**Note:** Not all raw data was used for estimation. The data actually used for estimation has been cropped and is available in the `estimates` Python instance.



Usage:

```Python
from src import *

c = LoadEstimates('data/...') # Name a .npy file
c.cropped_data # Cropped data
c.estimates_b # Frequency estimates
```

