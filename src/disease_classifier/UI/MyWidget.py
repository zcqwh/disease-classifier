
from PyQt5 import QtWidgets,QtCore

class MyTable(QtWidgets.QTableWidget):
    dropped = QtCore.pyqtSignal(list) 
    def __init__(self, *args, **kwargs):
        super(MyTable, self).__init__( *args, **kwargs)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        #self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.drag_item = None
        self.drag_row = None

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        self.drag_item = None
        if event.mimeData().hasUrls:
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            links = []
            
            for url in event.mimeData().urls():
                links.append(str(url.toLocalFile()))  
            self.dropped.emit(links) #发射信号
            
        else:
            event.ignore()       

    def startartDrag(self, supportedActions):
        super(MyTable, self).startDrag(supportedActions)
        self.drag_item = self.currentItem()
        self.drag_row = self.row(self.drag_item)


# =============================================================================
# class MyLineEdit(QtWidgets.QLineEdit):
#     dropped = QtCore.pyqtSignal(str) 
#     def __init__( self, *args, **kwargs):
#         super(MyLineEdit, self).__init__(*args, **kwargs)
#         self.setAcceptDrops(True)
#         self.setDragEnabled(True)
#     def dragEnterEvent(self, event):
#         if event.mimeData().hasUrls:
#             event.accept()
#         else:
#             event.ignore()
# 
#     def dragMoveEvent(self, event):
#         if event.mimeData().hasUrls:
#             event.setDropAction(QtCore.Qt.CopyAction)
#             event.accept()
#         else:
#             event.ignore()
# 
#     def dropEvent(self, event):
#         self.drag_item = None
#         if event.mimeData().hasUrls:
#             event.setDropAction(QtCore.Qt.CopyAction)
#             event.accept()
#             links = []
#             
#             for url in event.mimeData().urls():
#                 links.append(str(url.toLocalFile()))  
#             self.dropped.emit(links) #发射信号
#             
#         else:
#             event.ignore()       
# =============================================================================

        
class MyLineEdit(QtWidgets.QLineEdit):
    dropped = QtCore.pyqtSignal(str) 
    def __init__( self, *args, **kwargs):
        super(MyLineEdit, self).__init__(*args, **kwargs)
        self.setDragEnabled(True)

    def dragEnterEvent( self, event):
        data = event.mimeData()
        urls = data.urls()
        if ( urls and urls[0].scheme() == 'file' ):
            event.acceptProposedAction()

    def dragMoveEvent( self, event ):
        data = event.mimeData()
        urls = data.urls()
        if ( urls and urls[0].scheme() == 'file' ):
            event.acceptProposedAction()

    def dropEvent( self, event ):
        data = event.mimeData()
        urls = data.urls()
        if ( urls and urls[0].scheme() == 'file' ):
            # for some reason, this doubles up the intro slash
            filepath = str(urls[0].path())[1:]
            self.setText(filepath)
        self.dropped.emit(filepath)

