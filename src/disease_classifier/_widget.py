"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""


from qtpy import QtCore, QtGui, QtWidgets, uic
from .UI import art_rc
import os,sys,h5py,pickle,shutil
import cv2
import time
import numpy as np
import pandas as pd

from pytranskit.optrans.continuous.cdt import CDT
from statsmodels.nonparametric.kernel_density import KDEMultivariate

from . import aid_cv2_dnn
from .background_program import bin_2_rtdc


dir_root = os.path.dirname(__file__)#ask the module for its origin



class Disease_classifier(QtWidgets.QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        
        sys.path.append(os.path.join(dir_root,"UI")) # For finding CustomWidget
        uic.loadUi(os.path.join(dir_root,"UI","disease_classifier.ui"),self)
        
# =============================================================================
#         Link functions
# =============================================================================
        ######## table_aid_files########
        for i in [0,1,3]:
            self.table_aid_files.horizontalHeader().setSectionResizeMode(i,QtWidgets.QHeaderView.ResizeToContents)
        self.table_aid_files.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table_aid_files.setAcceptDrops(True)
        self.table_aid_files.setDragEnabled(True)
        
        self.table_aid_files.dropped.connect(self.dataDropped_aid)
        self.table_aid_files.horizontalHeader().sectionClicked.connect(self.select_all_aid)
        
        ######## table_aid_analy ########
        for i in [0,1,3,4]:
            self.table_aid_analy.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        for i in [2]:
            self.table_aid_analy.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)
        self.table_aid_analy.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        
        self.table_aid_analy.horizontalHeader().sectionClicked.connect(self.send_to_napari_all_class)
        
        ######## table_dise_class ########
        self.table_dise_class.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table_dise_class.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        

        self.lineEdit_aid_model_path.dropped.connect(self.show_model_choose)
        self.btn_aid_load_model.clicked.connect(self.aid_load_model)
        
        self.btn_aid_classify.clicked.connect(self.run_classify)
        self.btn_aid_add_rtdc.clicked.connect(self.aid_add_rtdc)
    
    def dataDropped_aid(self, l):
        #Iterate over l and check if it is a folder or a file (directory)    
        isfile = [os.path.isfile(str(url)) for url in l]
        isfolder = [os.path.isdir(str(url)) for url in l]


        #####################For folders with rtdc or bin:##########################            
        #where are folders?
        ind_true = np.where(np.array(isfolder)==True)[0]
        foldernames = list(np.array(l)[ind_true]) #select the indices that are valid
        #On mac, there is a trailing / in case of folders; remove them
        foldernames = [os.path.normpath(url) for url in foldernames]

        basename = [os.path.basename(f) for f in foldernames]
        #Look quickly inside the folders and ask the user if he wants to convert
        #to .rtdc (might take a while!)
        if len(foldernames)>0: #User dropped (also) folders (which may contain images)
            url_converted = []
            for url in foldernames:
                #get a list of tiff files inside this directory:
                images = []
                for root, dirs, files in os.walk(url):
                    for file in files:
                        if file.endswith(".rtdc"):
                            url_converted.append(os.path.join(root, file)) 
                        if file.endswith(".bin"):
                            url_converted.append(os.path.join(root, file)) 
                                  
            self.dataDropped_aid(url_converted)

        #####################For .rtdc or .bin files:##################################            
        #where are files?
        ind_true = np.where(np.array(isfile)==True)[0]
        filenames = list(np.array(l)[ind_true]) #select the indices that are valid
        filenames = [x for x in filenames if x.endswith(".rtdc") or x.endswith(".bin")]
        
        fileinfo = []
        for file_path in filenames:
                fileinfo.append({"file_path":file_path})
        
        
        for rowNumber in range(len(fileinfo)):#for url in l:
            url = fileinfo[rowNumber]["file_path"]

            #add to table
            rowPosition = self.table_aid_files.rowCount()
            self.table_aid_files.insertRow(rowPosition)

            columnPosition = 0
            #for each item, also create 2 checkboxes (train/valid)
            item = QtWidgets.QTableWidgetItem()#("item {0} {1}".format(rowNumber, columnNumber))
            item.setFlags( QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled  )
            item.setCheckState(QtCore.Qt.Unchecked)
            self.table_aid_files.setItem(rowPosition, columnPosition, item)

            columnPosition = 1
            #Place a button which allows to send to napari for viewing
            btn = QtWidgets.QPushButton(self.table_aid_files)
            btn.setObjectName("btn_load")
            btn.setMinimumSize(QtCore.QSize(30, 30))
            btn.setMaximumSize(QtCore.QSize(100, 100))
            icon_eye = QtGui.QIcon()
            icon_eye.addPixmap(QtGui.QPixmap(":/icon/eye.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            btn.setIcon(icon_eye)
            self.table_aid_files.setCellWidget(rowPosition, columnPosition, btn) 
            self.table_aid_files.resizeRowsToContents()
            btn.clicked.connect(self.load_rtdc_images)

            columnPosition = 2
            line = QtWidgets.QTableWidgetItem(str(url)) 
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_files.setItem(rowPosition,columnPosition, line) 
            
            columnPosition = 3
            btn_delete = QtWidgets.QPushButton(self.table_aid_files)
            
            btn_delete.setMinimumSize(QtCore.QSize(30, 30))
            btn_delete.setMaximumSize(QtCore.QSize(100, 100))
            icon_del = QtGui.QIcon()
            icon_del.addPixmap(QtGui.QPixmap(":/icon/delete.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            btn_delete.setIcon(icon_del)
            self.table_aid_files.setCellWidget(rowPosition, columnPosition, btn_delete) 
            self.table_aid_files.resizeRowsToContents()
            btn_delete.clicked.connect(self.delete_item)        

    def select_all_aid(self,col):
        """
        Check/Uncheck items on table_dragdrop
        """
        parent = self.sender().parent()
        if col == 0:
            rows = range(parent.rowCount()) #Number of rows of the table
            tableitems = [parent.item(row, col) for row in rows]
            
            checkStates = [tableitem.checkState() for tableitem in tableitems]
            checked = [state==QtCore.Qt.Checked for state in checkStates]
            if set(checked)=={True}:#all are checked!
                #Uncheck all!
                for tableitem in tableitems:
                    tableitem.setCheckState(QtCore.Qt.Unchecked)
            else:#otherwise check all   
                for tableitem in tableitems:
                    tableitem.setCheckState(QtCore.Qt.Checked)
                    
        if col == 1: #load all
            children= parent.findChildren(QtWidgets.QPushButton,"btn_load")
            for button in children:
                button.click() #click load button

        if col == 3: #delete all
            for i in range(parent.rowCount()):
                parent.removeRow(0)
    
    def delete_item(self,item):
        """
        delete table item and corresponding layers
        """
        buttonClicked = self.sender()
        table = buttonClicked.parent().parent()

        index = table.indexAt(buttonClicked.pos())
        rowPosition = index.row()
        table.removeRow(rowPosition) #remove table item
    
    
    def load_rtdc_images(self):
        buttonClicked = self.sender()
        index = self.table_aid_files.indexAt(buttonClicked.pos())
        rowPosition = index.row()
    
        file_path = self.table_aid_files.item(rowPosition, 2).text()
        
        if file_path.endswith(".rtdc"):
            rtdc_ds = h5py.File(file_path,"r")
    
            #Get the images from .rtdc file
            images = np.array(rtdc_ds["events"]["image"]) 
        
            name = os.path.basename(file_path)
            new_layer = self.viewer.add_image(images,name=name)
        
        if file_path.endswith(".bin"):
            binary = np.fromfile(file_path, dtype='>H')
            #Get the images from .bin
            n,w,h = binary[1],binary[3],binary[5]
            images = binary[6:].reshape(n,h,w)
            
            name = os.path.basename(file_path)[:-4] #remove .bin
            new_layer = self.viewer.add_image(images,name=name)
            
        
    
    def aid_load_model(self):
        model_path = QtWidgets.QFileDialog.getExistingDirectory(self)
        if model_path:
            self.lineEdit_aid_model_path.setText(model_path)
        
    
    def show_model_choose(self,model_path):
        self.lineEdit_aid_model_path.setText(model_path)
        #Search models in the folder 
        dir_list = []
        for path, dir_, file in os.walk("/"+model_path,"r"):
            for dir_name in dir_:
                dir_list.append(dir_name)
        
        # Check if CNN morphological classification exists
        if "01_model" in dir_list:
            self.label_phenotype.setText("01_model")
        else:    
            self.popup_prompt("Could not find CNN morphological classification model.")
        
        # Check if rf/plda disease prediction model exidts
        if "02_rf_model"in dir_list and "03_cdt_plda_models" in dir_list:
            self.popup_prompt("Multiple models are chosen.")
        elif "02_rf_model" in dir_list:
            self.label_disease.setText("02_rf_model")
        elif "03_cdt_plda_models" in dir_list:
            self.label_disease.setText("03_cdt_plda_models")
        else:
            self.popup_prompt("Model not found.")


    def aid_classify_rtdc(self,model_path,rtdc_path):
        
        #Load .rtdc
        rtdc_ds = h5py.File(rtdc_path,"r")
        images = np.array(rtdc_ds["events"]["image"]) # get the images
        pos_x, pos_y = rtdc_ds["events"]["pos_x"][:], rtdc_ds["events"]["pos_y"][:] 
        pix = rtdc_ds.attrs["imaging:pixel size"] # pixelation (um/pix)
        
        ############### CNN ############### 
        #Load the CNN morphological classification model
        meta_path = os.path.join(model_path,"01_model","M10_Nitta6l_32pix_8class_meta.xlsx") 
        img_processing_settings = aid_cv2_dnn.load_model_meta(meta_path) # Extract image preprocessing settings from meta file
        model_pb_path = os.path.join(model_path,"01_model","M10_Nitta6l_32pix_8class_448_optimized.pb") 
        model_pb = cv2.dnn.readNet(model_pb_path)

        # Compute the predictions
        scores = aid_cv2_dnn.forward_images_cv2(model_pb,img_processing_settings,images,pos_x,pos_y,pix)
        prediction = np.argmax(scores,axis=1)
        ############### end of CNN ############### 
        
        
        ############### Get the disease prediction ############### 
        rtdc_ds_len = rtdc_ds["events"]["image"].shape[0] #this way is actually faster than asking any other feature for its len :)
        prediction_fillnan = np.full([rtdc_ds_len], np.nan)#put initially np.nan for all cells

        classes = scores.shape[1]
        if classes>9:
            classes = 9#set the max number of classes to 9. It cannot saved more to .rtdc due to limitation of userdef

        #Make sure the predictions get again to the same length as the initial data set
        #Fill array with corresponding predictions
        index = range(len(images))
        for i in range(len(prediction)):
            indx = index[i]
            prediction_fillnan[indx] = prediction[i]
        #Predictions are integers
        prediction_fillnan = prediction_fillnan.astype(int)
        ############### Get the disease prediction ############### 


        ############### RF ############### 
        def RF_prediction():
            
            def p20(x):
                return np.percentile(x,20)
            def p50(x):
                return np.percentile(x,50)
            def p80(x):
                return np.percentile(x,80)

            def mad(x):#definition of median absolute deviation
                return np.median(abs(x - np.median(x)))/0.67448975019608171

            def getdata2(rtdc_path,userdef0):
                feature_name = ["Area","Area_Ratio"]
                keys = ["area_um","area_ratio"]
                classes = [1,2]

                print(rtdc_path)
                NameList,List = [],[]
                
                rtdc_ds = h5py.File(rtdc_path, 'r')

                operations = [np.mean,np.median,np.std,mad,p20,p50,p80]
                operationnames = ["mean","median","std","mad","p20","p50","p80"]

                #get the numbers of events for each subpopulation
                #userdef0 = rtdc_ds["events"]["userdef0"][:]

                for cl in classes:
                    ind_x = np.where(userdef0==cl)[0]
                    perc = len(ind_x)/len(userdef0)
                    NameList.append("events_perc_class_"+str(cl))
                    List.append(perc)
                    
                    for k in range(len(keys)):
                        values = rtdc_ds["events"][keys[k]][:][ind_x]
                        #remove nan values and zeros
                        ind = np.isnan(values)
                        ind = np.where(ind==False)[0]
                        values = values[ind]
                        ind = np.where(values!=0)[0]
                        values = values[ind]
                
                        for o in range(len(operations)):
                            NameList.append(feature_name[k]+"_"+operationnames[o]+"_class"+str(cl))
                            if len(values)==0:
                                List.append(np.nan)
                            else:
                                stat = operations[o](values)
                                List.append(stat)

                return [List,NameList]


            #Get area, area_ratio values for each cell type
            X_features = getdata2(rtdc_path,prediction_fillnan)
            values = np.array(X_features[0])
            values = values.reshape(-1,values.shape[0])
            FeatNames = X_features[1]
            X_features = pd.DataFrame(values,columns=FeatNames)

            #re-order the features (Random forest expexts a certain order)
            features = []
            features.append("events_perc_class_1")
            features.append("events_perc_class_2")

            for i in ["1","2"]:
                features.append("Area_mean_class"+i)
                features.append("Area_std_class"+i)
                features.append("Area_mad_class"+i)
                features.append("Area_p20_class"+i)
                features.append("Area_p50_class"+i)
                features.append("Area_p80_class"+i)

                features.append("Area_Ratio_mean_class"+i)
                features.append("Area_Ratio_std_class"+i)
                features.append("Area_Ratio_mad_class"+i)
                features.append("Area_Ratio_p20_class"+i)
                features.append("Area_Ratio_p50_class"+i)
                features.append("Area_Ratio_p80_class"+i)

            #Disease classification using Rndom forest model
            rf_model_path = os.path.join(model_path,"02_rf_model","09_RF_v02.3.sav")
            #load the random forest model
            rf_model = pickle.load(open(rf_model_path, 'rb'))

            X_features = X_features[features]
            disease_probab = rf_model.predict_proba(X_features)

            #print(f"disease prediction:{disease_probab}")
            return disease_probab



            
        
        ############### CDT-PCA-PLDA ############### 
        def PLDA_prediction():
            def getdata3(rtdc_path,userdef0):
                """
                Get distributions: 
                    - area and solidity for platelets
                    - area and solidity for clusters
                """
                feature_name = ["Area","Solidity"]
                keys = ["area_um","area_ratio"]
                classes = [1,2] #1=single platelet, 2=cluster

                print(rtdc_path)
                NameList,List = [],[]
                
                rtdc_ds = h5py.File(rtdc_path, 'r')


                for cl in classes:
                    ind_x = np.where(userdef0==cl)[0]        
                    for k in range(len(keys)):
                        values = rtdc_ds["events"][keys[k]][:][ind_x]
                        #remove nan values and zeros
                        ind = np.isnan(values)
                        ind = np.where(ind==False)[0]
                        values = values[ind]
                        ind = np.where(values!=0)[0]
                        values = values[ind]
                        if keys[k]=="area_ratio":
                            values = 1/values#convert to solidity (solidity=1/area_ratio)
                        
                        NameList.append(feature_name[k]+"_class"+str(cl))
                        if len(values)==0:
                            List.append(np.nan)
                        else:
                            List.append(values)

                return [List,NameList]

            model_pca_path = os.path.join(model_path,"03_cdt_plda_models","02_PCA_v06.4.sav")
            model_plda_path = os.path.join(model_path,"03_cdt_plda_models","02_PLDA_v06.4.sav")
            norm_path = os.path.join(model_path,"03_cdt_plda_models","02_PLDA_v06_norm.csv") # normalization parameters
            
            DF_norm = pd.read_csv(norm_path)
            mn = DF_norm["mn"]
            av = DF_norm["av"]
            sd = DF_norm["sd"]
            
            X = getdata3(rtdc_path,prediction_fillnan)
            values = np.array(X[0])
            values = values.reshape(-1,values.shape[0])
            FeatNames = X[1]
            X = pd.DataFrame(values,columns=FeatNames)
            X.reset_index(drop=True, inplace=True)


            mxMP = 1000
            Ldst = 5000
            sprd = 25
            d_dom=np.linspace(0,mxMP,num=Ldst)
            d_dom=np.expand_dims(d_dom,axis=1)



            #Transform data into density using KDE
            X_kde = np.zeros((X.shape[0],Ldst,X.shape[1]))
            dX = pd.DataFrame(X)
            for a in range(X.shape[0]):
                for b in range(X.shape[1]):
                    tmp = dX.iat[a,b]
                    tmp = tmp-mn[b]
                    t = av[b]+sprd*sd[b]
                    tmp = mxMP*tmp/t
                    tmp = np.expand_dims(tmp,axis=1)
                    
                    kde = KDEMultivariate(tmp,var_type='c')
                    pdf = kde.pdf(d_dom)
                    pdf = 100*pdf/pdf.sum()
                    
                    X_kde[a,:,b] = pdf
                        
                print('kde-p'+str(a+1))


            def cdt_transform(X):
                cdt = CDT()

                epsilon = 1e-7
                N = X.shape[1]
                x0 = np.linspace(0, 1, N)
                x = np.linspace(0, 1, N)
                I0 = np.ones(x0.size)
                I0 = abs(I0) + epsilon
                I0 = I0/I0.sum()
                
                X_hat = np.zeros((X.shape[0],X.shape[1]-2,X.shape[2]))
                for a in range(X.shape[0]):
                    for b in range(X.shape[2]):
                        I1 = X[a,:,b]
                        I1 = abs(I1) + epsilon
                        I1 = I1/I1.sum()
                        I1_hat, I1_hat_old, xtilde = cdt.forward(x0, I0, x, I1,rm_edge=True)
                        X_hat[a,:,b]=I1_hat
                
                X_hat = X_hat.reshape(X_hat.shape[0],-1)
                return X_hat


            #load the PCA and PLDA models
            plda_model = pickle.load(open(model_plda_path, 'rb'))
            pca_model = pickle.load(open(model_pca_path, 'rb'))


            X_cdt = cdt_transform(X_kde)
            X_pca  = pca_model.transform(X_cdt)
            disease_probab = plda_model.predict_proba(X_pca)
            print(f"prediction scores [thromb/covid]:{scores[0]}/{scores[1]}")
            
            return np.reshape(disease_probab,(1,2))
        
        
        if self.label_disease.text() == "02_rf_model":
            disease_probab = RF_prediction()
        if self.label_disease.text() == "03_cdt_plda_models":
            disease_probab = PLDA_prediction()
        
        return prediction, images, (pos_x/pix).astype(int), (pos_y/pix).astype(int), scores, model_pb_path, disease_probab
    
    
        
    
    
    def show_results(self,prediction,disease_probab):
        #Statistics
        result = pd.value_counts(prediction).sort_index()
        percentage = pd.value_counts(prediction,normalize=True).sort_index()
        
        indexs = [index for index,v in result.items()]
        values = [value for i,value in result.items()]
        pcts = [pct for i,pct in percentage.items()]
        Name = ["Noise","single Platelet", "multiple Platelets","single WBC", 
                "single WBC+Platelets", "multiple WBC", "multiple WBC+Platelets" ]
        
        #clear content before add new
        if not self.table_aid_analy.rowCount() ==0:
            for i in range(self.table_aid_analy.rowCount()):
                self.table_aid_analy.removeRow(0) 
        
        
        for rowNumber in range(len(result)):
            index = indexs[rowNumber]
            value = values[rowNumber]
            pct = np.round(pcts[rowNumber],4)
            
            #add to table
            rowPosition = self.table_aid_analy.rowCount()
            self.table_aid_analy.insertRow(rowPosition)
            
            columnPosition = 0 # load
            #Place a button which allows to send to napari for viewing
            btn = QtWidgets.QPushButton(self.table_aid_analy)
            btn.setObjectName("btn_load")
            btn.setMinimumSize(QtCore.QSize(30, 30))
            btn.setMaximumSize(QtCore.QSize(100, 100))
            
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/icon/visibility_off.png"), QtGui.QIcon.Active, QtGui.QIcon.On)
            icon.addPixmap(QtGui.QPixmap(":/icon/eye.png"), QtGui.QIcon.Active, QtGui.QIcon.Off)
            btn.setIcon(icon)
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.toggle()
            btn.clicked.connect(self.send_to_napari_pred_class)
            
            self.table_aid_analy.setCellWidget(rowPosition, columnPosition, btn) 
            self.table_aid_analy.resizeRowsToContents()
            
            columnPosition = 1 # class column
            line = QtWidgets.QTableWidgetItem(str(index)) 
            
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_analy.setItem(rowPosition,columnPosition, line) 
            
            columnPosition = 2 # name
            line = QtWidgets.QTableWidgetItem(Name[index]) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_analy.setItem(rowPosition,columnPosition, line) 
            
            
            columnPosition = 3 # number
            line = QtWidgets.QTableWidgetItem(str(value)) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_analy.setItem(rowPosition,columnPosition, line) 
            
            columnPosition = 4 # percentage
            line = QtWidgets.QTableWidgetItem(str(pct)) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_analy.setItem(rowPosition,columnPosition, line) 
            
            pro_bar = QtWidgets.QProgressBar(self.table_aid_analy)
            pro_bar.setRange(0, 100)
            pro_bar.setValue(int(pct*100))
            self.table_aid_analy.setCellWidget(rowPosition, columnPosition, pro_bar) 
        
        
        #clear content before add new
        if not self.table_dise_class.rowCount() ==0:
            for i in range(self.table_dise_class.rowCount()):
                self.table_dise_class.removeRow(0) 
        
        Name = ["Thrombosis" , "COVID-19"]
        for rowNumber in range(2):
            #add to table
            rowPosition = self.table_dise_class.rowCount()
            self.table_dise_class.insertRow(rowPosition)
            
            columnPosition = 0 # disease name
            line = QtWidgets.QTableWidgetItem(Name[rowNumber]) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_dise_class.setItem(rowPosition,columnPosition, line) 
            
            columnPosition = 1 # percentage
            probability = disease_probab[0][rowNumber]
            line = QtWidgets.QTableWidgetItem(probability) 
            line.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_dise_class.setItem(rowPosition,columnPosition, line) 
            
            pro_bar = QtWidgets.QProgressBar(self.table_dise_class)
            n = 100

            # setting maximum value for 2 decimal points
            pro_bar.setMaximum(100 * n)
            
            # value in percentage
            value = probability*100
            
            # setting the value by multiplying it to 100
            pro_bar.setValue(value * n)
            
            # displaying the decimal value
            pro_bar.setFormat("%.02f %%" % value)
            self.table_dise_class.setCellWidget(rowPosition, columnPosition, pro_bar) 
            
            
            
    def run_classify(self):
        rtdc_paths = []
        bin_paths = []
        rowPosition = self.table_aid_files.rowCount()
        for i in range(rowPosition): # read all checked data
            if self.table_aid_files.item(i, 0).checkState(): #return 0 or 2,,0:uncheck, 2:checked
                file_path = self.table_aid_files.item(i, 2).text()
                if file_path.endswith('.rtdc'):
                    rtdc_paths.append(file_path)
                if file_path.endswith('.bin'):
                    bin_paths.append(file_path)
        
        if rtdc_paths==[] and bin_paths == []:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("No .rtdc or .bin selected.")
            x = msg.exec_()
        
        if self.lineEdit_aid_model_path.text()=='':
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("Error")
            msg.setText("No model selected.")
            x = msg.exec_()
            
        else:
            model_path = "/" + self.lineEdit_aid_model_path.text() 
        
            if len(rtdc_paths)>0:
                for rtdc_path in rtdc_paths:
                    self.prediction, self.images, self.pos_x, self.pos_y, scores, model_pb_path, disease_probab = self.aid_classify_rtdc(model_path,rtdc_path)
                    self.show_results(self.prediction,disease_probab)
            
            if len(bin_paths)>0:
                for bin_path in bin_paths:
                    rtdc_path = bin_2_rtdc(bin_path)
                    self.prediction, self.images, self.pos_x, self.pos_y, scores, model_pb_path, disease_probab = self.aid_classify_rtdc(model_path,rtdc_path)
                    self.show_results(self.prediction,disease_probab)
                        

    
    
    def send_to_napari_pred_class(self):
        #get class images
        buttonClicked = self.sender()
        index = self.table_aid_analy.indexAt(buttonClicked.pos())
        rowPosition = index.row()
        class_num = self.table_aid_analy.item(rowPosition,1).text()
        images = self.images
        class_ind = np.where(self.prediction==int(class_num))
        class_images = images[class_ind]
        class_name = "Class " + class_num
        
        #add class label
        x_ = self.pos_x[class_ind]
        y_ = self.pos_y[class_ind]
        label = []
        for i in range(len(class_images)):
            mask = np.zeros((67,67,4),dtype=np.uint8) 
            x = x_[i]
            y = y_[i] 
            cv2.circle(mask, (x,y), 1, (0,255,0,255),-1)
            cv2.putText(mask,class_num,(2,11),cv2.FONT_HERSHEY_DUPLEX, 0.4, (255,255,255,255),1)
            label.append(mask)
        label = np.array(label)
        label_name = "Label " + class_num
        
        #button status
        if buttonClicked.isChecked(): #add layer
            new_layer = self.viewer.add_image(class_images,name=class_name)
            lable_layer = self.viewer.add_image(label, name=label_name)     
        else: #delete layer
            existing_layers = {layer.name for layer in self.viewer.layers} 
            if class_name in existing_layers:
                self.viewer.layers.remove(class_name)
            if label_name in existing_layers:
                self.viewer.layers.remove(label_name)
            
        
    def send_to_napari_all_class(self,col):
        if col==0:
            parent = self.sender().parent()
            children= parent.findChildren(QtWidgets.QPushButton,"btn_load")
            for button in children:
                button.click() #click load button


    def aid_add_rtdc(self):
        model_path = "/" + self.lineEdit_aid_model_path.text() 
        rowPosition = self.table_aid_files.rowCount()
        file_paths = []
        save_paths = []
        for i in range(rowPosition): # read all data
            #if self.table_aid_files.item(i, 0).checkState(): #return 0 or 2,,0:uncheck, 2:checked
                file_path = self.table_aid_files.item(i, 2).text()
                file_paths.append(file_path)
        
        for file_path in file_paths:
            if file_path.endswith('.rtdc'):
                rtdc_path = file_path
            if file_path.endswith('.bin'):
                dic = bin_2_rtdc(file_path)
                
            prediction, images, pos_x, pos_y, scores, model_pb_path,disease_probab = self.aid_classify_rtdc(model_path,rtdc_path)
            rtdc_ds = h5py.File(rtdc_path,"r")
            
            ###################append scores and pred to .rtdc file########################
            rtdc_ds_len = rtdc_ds["events"]["image"].shape[0] #this way is actually faster than asking any other feature for its len :)
            prediction_fillnan = np.full([rtdc_ds_len], np.nan)#put initially np.nan for all cells
            
            classes = scores.shape[1]
            if classes>9:
                classes = 9#set the max number of classes to 9. It cannot saved more to .rtdc due to limitation of userdef
            scores_fillnan = np.full([rtdc_ds_len,classes], np.nan)
    
            #Make sure the predictions get again to the same length as the initial data set
            #Fill array with corresponding predictions
            index = range(len(images))
    
            for i in range(len(prediction)):
                indx = index[i]
                prediction_fillnan[indx] = prediction[i]
                #if export_option == "Append to .rtdc":
                #for class_ in range(classes):
                scores_fillnan[indx,0:classes] = scores[i,0:classes]
    
            #Get savename
            path, rtdc_file = os.path.split(rtdc_path)
            
            fname_addon = os.path.split(model_pb_path)[-1]#the filename of the model
            fname_addon = fname_addon.split(".pb")[0]
            fname_addon = self.anti_vowel(fname_addon)#remove the vowels to make it shorter
            savename = rtdc_path.split(".rtdc")[0]
            savename = savename+"_"+str(fname_addon)+".rtdc"
            save_paths.append(savename)
            
            if not os.path.isfile(savename):#if such a file does not yet exist...
                savename = savename
            else:#such a file already exists!!!
                #Avoid to overwriting an existing file:
                print("Adding additional number since file exists!")
                addon = 1
                while os.path.isfile(savename):
                    savename = savename.split(".rtdc")[0]
                    if addon>1:
                        savename = savename.split("_"+str(addon-1))[0]
                    savename = savename+"_"+str(addon)+".rtdc"
                    addon += 1        
    
            #print(f"Save as {savename}")                    
            shutil.copy(rtdc_path, savename) #copy original file
            #append to hdf5 file
            with h5py.File(savename, mode="a") as h5:
                h5["events/userdef0"] = prediction_fillnan
                #add the scores to userdef1...9
                userdef_ind = 1
                for class_ in range(classes):
                    scores_i = scores_fillnan[:,class_]
                    h5["events/userdef"+str(userdef_ind)] = scores_i
                    userdef_ind += 1
            
            
        ###################### show messagebox ######## 
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("")
        text = "\n".join(save_paths)
        msg.setText("Successfully export .rtdc files in \n" + text)
        
        x = msg.exec_()

    def anti_vowel(self,c):
        newstr = ""
        vowels = ('a', 'e', 'i', 'o', 'u','A', 'E', 'I', 'O', 'U')
        for x in c.lower():
            if x in vowels:
                newstr = ''.join([l for l in c if l not in vowels])    
        return newstr
    
    def popup_prompt(self,s):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle("Error")
        msg.setText(s)
        x = msg.exec_()
        
        
        
        
