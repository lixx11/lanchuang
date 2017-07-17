#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
from os.path import dirname, join, abspath

from PyQt4 import QtCore
from PyQt4 import QtGui
from PyQt4 import uic
import pyqtgraph as pg
from pyqtgraph import PlotDataItem, PlotItem
from pyqtgraph.parametertree import ParameterTree, Parameter

import numpy as np


class Evaluator(QtGui.QMainWindow):
  """docstring for CXIWindow"""
  def __init__(self):
    super(Evaluator, self).__init__()
    # load and adjust layout
    dir_ = os.path.dirname(os.path.abspath(__file__))
    uic.loadUi(dir_ + '/' + 'layout.ui', self)
    self.splitterH.setSizes([self.width()*0.7, self.width()*0.3])
    self.splitterV.setSizes([self.height()*0.7, self.height()*0.3])

    self.labelItem = PlotDataItem(pen=pg.mkPen('y', width=1, 
      style=QtCore.Qt.SolidLine), name='labels')
    self.probItem = PlotDataItem(pen=pg.mkPen('c', width=1,
      style=QtCore.Qt.SolidLine), name='prob')
    self.evalPlot.addItem(self.labelItem)
    self.evalPlot.addItem(self.probItem)
    
    # initilize some parameters
    self.frame_id = 0

    # setup menu slots
    self.actionOpenNpz.triggered.connect(self.loadNpz)
    self.actionOpenModel.triggered.connect(self.loadModel)

     # setup parameter tree
    params_list = [
            {'name': u'文件信息', 'type': 'group', 'children': [
              {'name': u'路径', 'type': 'str', 'value': u'未设置', 'readonly': True},
              {'name': u'帧数', 'type': 'str', 'value': u'未设置', 'readonly': True},
              {'name': u'长X高', 'type': 'str', 'value': u'未设置', 'readonly': True},
              {'name': u'模型', 'type': 'str', 'value': u'未设置', 'readonly': True},
            ]},
            {'name': u'基本操作', 'type': 'group', 'children': [
              {'name': u'当前帧', 'type': 'int', 'value': self.frame_id},
              {'name': u'开始测试', 'type': 'action'},
            ]},
            ]
    self.params = Parameter.create(name='params', type='group', children=params_list)
    self.parameterTree.setParameters(self.params, showTop=False)


    # parameter connection
    self.params.param(u'基本操作', 
      u'当前帧').sigValueChanged.connect(self.frameChangedSlot)
    self.params.param(u'基本操作', 
      u'开始测试').sigActivated.connect(self.eval)

  def eval(self):
    cnn_dir = join(dirname(dirname(abspath(__file__))), 'cnn')
    sys.path.insert(0, cnn_dir)
    import cnn_eval
    ckpt_file = self.model_file[:-5]

    probs = cnn_eval.eval(data_file=self.data_file,
                          ckpt_file=ckpt_file,
                          crop_size=(100, 270))
    self.probs = probs
    self.probItem.setData(probs[:,1])
    
  def loadModel(self):
    fpath = str(QtGui.QFileDialog.getOpenFileName(self, 
      '打开文件', '', 'META File (*.meta)'))
    self.model_file = fpath
    self.params.param(u'文件信息', u'模型').setValue(fpath)

  def frameChangedSlot(self, _, frame_id):
    self.frame_id = int(frame_id)
    self.updateDisp()

  def loadNpz(self):
    fpath = str(QtGui.QFileDialog.getOpenFileName(self, 
      '打开文件', '', 'NPZ File (*.npz)'))
    self.data_file = fpath
    self.data = np.load(fpath)

    self.params.param(u'文件信息', u'路径').setValue(fpath)
    shape = self.data['frames'].shape
    self.params.param(u'文件信息', u'帧数').setValue(shape[0])
    self.params.param(u'文件信息', u'长X高').setValue('%dX%d' % 
      (shape[2], shape[1]))
    self.updateDisp()
    self.updatePlot()

  def updateDisp(self):
    image = self.data['frames'][self.frame_id].T
    self.imageView.setImage(image, 
      autoRange=False, autoLevels=False, autoHistogramRange=False)

  def updatePlot(self):
    labels = self.data['labels']
    self.labelItem.setData(labels)


if __name__ == '__main__':
  app = QtGui.QApplication(sys.argv)
  win = Evaluator()
  win.setWindowTitle("Evaluator")
  win.show()
  app.exec_()