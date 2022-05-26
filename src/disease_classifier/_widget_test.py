"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/plugins/stable/guides.html#widgets

Replace code below according to your needs.
"""
import napari
from PyQt5 import QtCore,QtGui,QtWidgets,uic
import numpy as np
import sys,os


sys.path.append("./UI")
#import art_rc

class Disease_classifier(QtWidgets.QWidget):
    def __init__(self):
        super(Disease_classifier,self).__init__()
        
        #Load the ui file
        uic.loadUi("./UI/disease_classifier.ui",self)
        
        
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
        
        ######## table_aid_analy ########
        for i in [0,1,3,4]:
            self.table_aid_analy.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.ResizeToContents)
        for i in [2]:
            self.table_aid_analy.horizontalHeader().setSectionResizeMode(i, QtWidgets.QHeaderView.Stretch)
        
        ######## table_dise_class ########
        self.table_dise_class.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        
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
# =============================================================================
#             btn.clicked.connect(self.load_rtdc_images)
# =============================================================================
            
            icon = QtGui.QIcon()
            icon.addPixmap(QtGui.QPixmap(":/icon/eye.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            btn.setIcon(icon)
            self.table_aid_files.setCellWidget(rowPosition, columnPosition, btn) 
            self.table_aid_files.resizeRowsToContents()

            columnPosition = 2
            line = QtWidgets.QTableWidgetItem(str(url)) 
            line.setFlags( QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
            self.table_aid_files.setItem(rowPosition,columnPosition, line) 
            
            columnPosition = 3
            btn_delete = QtWidgets.QPushButton(self.table_aid_files)
            
            btn_delete.setMinimumSize(QtCore.QSize(30, 30))
            btn_delete.setMaximumSize(QtCore.QSize(100, 100))
            icon_2 = QtGui.QIcon()
            icon_2.addPixmap(QtGui.QPixmap(":/icon/delete.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            btn_delete.setIcon(icon_2)
            
            
            self.table_aid_files.setCellWidget(rowPosition, columnPosition, btn_delete) 
            self.table_aid_files.resizeRowsToContents()
# =============================================================================
#             btn_delete.clicked.connect(self.delete_item)        
# =============================================================================


#Show napari viewer
viewer = napari.Viewer()
#Add widget to napari viewer
viewer.window.add_dock_widget(Disease_classifier(),area="right",name="iacs_ipac_reader")