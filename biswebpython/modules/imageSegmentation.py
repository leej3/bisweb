# LICENSE
#
# _This file is Copyright 2018 by the Image Processing and Analysis Group (BioImage Suite Team). Dept. of Radiology & Biomedical Imaging, Yale School of Medicine._
#
# BioImage Suite Web is licensed under the Apache License, Version 2.0 (the "License");
#
# - you may not use this software except in compliance with the License.
# - You may obtain a copy of the License at [http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
#
# __Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.__
#
# ENDLICENSE



import sys
import json
try:
    import bisweb_path;
except ImportError:
    bisweb_path=0;

import biswebpython.core.bis_basemodule as bis_basemodule;
import biswebpython.core.bis_objects as bis_objects;
# import biswebpython.utilities.bidsUtils as bids_utils;
import biswebpython.utilities.imgSegm as imgSegm;



class imageSegmentation(bis_basemodule.baseModule):

    def __init__(self):
        super().__init__();
        self.name='imageSegmentation';

    def createDescription(self):
        return {
            "name": "imageSegmentation",
            "description": "Image Segmentation Using Deep Learning.",
            "version": "1.0",
            "inputs": [

            ],
            "outputs": [
            ],
            "params": [
                {
                    "name": "Debug",
                    "description": "Toggles debug logging. ",
                    "varname": "debug",
                    "type": "boolean",
                    "default": True
                },
                {
                    "name": "Jobfile",
                    "description": "User-defined parameter file for deep learning.",
                    "varname": "jobfile",
                    "shortname": "jf",
                    "type": "string",
                    "required": True,
                    "default": ""
                },
                {
                    "name": "Show Jobfile Example",
                    "description": 'Below is the example format of the input jobfile. NOTE: the argument: paramfile is not included in the jobfile example.',
                    "varname": "**********************************",
                    "type": "string",
                    "required": False,
                    "default": ""
                },
                {
                    "name": "Show Jobfile Example",
                    "description": 'path of input csv file',
                    "varname": "inputpath",
                    "type": "string",
                    "required": False,
                    "default": ""
                },
                {
                    "name": "Show Jobfile Example",
                    "description": 'path of output folder',
                    "varname": "outputmodelpath",
                    "type": "string",
                    "required": False,
                    "default": ""
                },
                {
                    "name": "Show Jobfile Example",
                    "description": 'the GPU ID you want to use.',
                    "varname": "gpu_device",
                    "type": "string",
                    "required": False,
                    "default": ""
                },
                {
                    "name": "Show Jobfile Example",
                    "description": 'The transformation that will be default applied to your training, validation and testing data. \
                                    The patch_size and spacing must be specified here.',
                    "varname": "defaulttransformation",
                    "type": "string",
                    "required": False,
                    "default": ""
                }

            ]
        }


    def directInvokeAlgorithm(self,vals):

        with open(vals['jobfile']) as jf:
            jfile = json.load(jf)
            if jfile['module'] != self.name:
                print ("Double check if the module name of the jobfile corrects!")
                raise ValueError

        try:
            imgSegm.imageSegmentation(jfile['params'], vals['debug'])

        except:
            e = sys.exc_info()[0]
            print('---- Failed to invoke algorithm ----',e);
            return False


        return True



if __name__ == '__main__':
    import biswebpython.core.bis_commandline as bis_commandline;
    sys.exit(bis_commandline.loadParse(imageSegmentation(),sys.argv,False));
