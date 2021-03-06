import json
import math
import numpy
import os
import random
import unittest
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
import logging

#
# MultiMapper
#

class MultiMapper(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "MultiMapper" # TODO make this more human readable by adding spaces
    self.parent.categories = ["Quantification"]
    self.parent.dependencies = []
    self.parent.contributors = ["Steve Pieper (Isomics, Inc.)"] # replace with "Firstname Lastname (Organization)"
    self.parent.helpText = """Tools for creating parametric maps from multidimensional MRI
"""
    self.parent.helpText += self.getDefaultModuleDocumentationLink()
    self.parent.acknowledgementText = """
Supported in part by the Neuroimage Analysis Center (NAC) https://nac.spl.harvard.edu/

NAC is a Biomedical Technology Resource Center supported by the National Institute of Biomedical Imaging and Bioengineering (NIBIB) (P41 EB015902). It was supported by the National Center for Research Resources (NCRR) (P41 RR13218) through December 2011.
""" # replace with organization, grant and thanks.

#
# MultiMapperWidget
#

class MultiMapperWidget(ScriptedLoadableModuleWidget):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setup(self):
    ScriptedLoadableModuleWidget.setup(self)

    # Instantiate and connect widgets ...
    self.qtiDemoButton = qt.QPushButton("Run mdMRI Explorer")
    self.layout.addWidget(self.qtiDemoButton)
    self.qtiDemoButton.connect('clicked()', self.qtiDemo)


    #
    # Parameters Area
    #
    parametersCollapsibleButton = ctk.ctkCollapsibleButton()
    parametersCollapsibleButton.text = "Parameters"
    self.layout.addWidget(parametersCollapsibleButton)

    # Layout within the dummy collapsible button
    parametersFormLayout = qt.QFormLayout(parametersCollapsibleButton)


    # no UI right now

    # Add vertical spacer
    self.layout.addStretch(1)

    self.logic = MultiMapperLogic()

  def cleanup(self):
    pass

  def qtiDemo(self):
    self.mml = MultiMapperLogic()
    self.mml.parallelCoordinatesSegmentation()


#
# MultiMapperLogic
#

class MultiMapperLogic(ScriptedLoadableModuleLogic):
  """
  Methods to convert multiple mdMRI volumes into parametric maps.

  WIP: currently hard-coded for first example case


  For testing:

setup:
slicer.util.pip_install('sklearn')

slicer.util.reloadScriptedModule('MultiMapper'); import MultiMapper; mml = MultiMapper.MultiMapperLogic(); mml.tableFromExperiment(); mml.plotsFromTable()


slicer.util.reloadScriptedModule('MultiMapper'); import MultiMapper; mml = MultiMapper.MultiMapperLogic(); mml.mapFromCrosshair()

slicer.util.reloadScriptedModule('MultiMapper'); import MultiMapper; mml = MultiMapper.MultiMapperLogic(); mml.segmentWithKMeans()

slicer.util.reloadScriptedModule('MultiMapper'); import MultiMapper; mml = MultiMapper.MultiMapperLogic(); mml.parallelCoordinatesParCoords_RandomSample()

slicer.util.reloadScriptedModule('MultiMapper'); import MultiMapper; mml = MultiMapper.MultiMapperLogic(); mml.parallelCoordinatesSegmentation()



  """

  def __init__(self):

    # estimated from 2HG m/z at 129
    self.intensities = {
        "R1" :  100.,
        "R2" :   30.,
        "R4" : 1350.,
        "R5" :  210.,
    }
    self.observerObjectIDPairs = []

  def __del__(self):
    print("logic is destructing")
    for object_, id_ in self.observerObjectIDPairs:
      object_.RemoveObserver(id_)

  def standardizedDistance(self, labelPair):
    """calculate the pairwise distance between two sample
    vectors and standardize by the variance.  If the
    sample vector dimensions are uncorrelated, then
    this is the Mahalanobis distance.
    """
    samplePair = [self.samples[label] for label in labelPair]
    accumulation = 0
    for name in self.names:
      difference = samplePair[1][name] - samplePair[0][name]
      accumulation += (difference * difference) / (self.variances[name] * self.variances[name])
    return math.sqrt(accumulation)

  def standardizedDistanceBetweenIndices(self, index0, index1):
    """Return standardized distance at an index point in array space"""
    accumulation = 0
    for nodeName in self.nodes:
      if nodeName in self.arrays:
        a = self.arrays[nodeName]
        difference = a[index0] - a[index1]
        accumulation += (difference * difference) / (self.variances[nodeName] * self.variances[nodeName])
    return math.sqrt(accumulation)

  def volumeStatistics(self):
    self.nodes = slicer.util.getNodes('dtd_covariance_*')
    self.arrays = {}
    self.means = {}
    self.variances = {}
    for nodeName in self.nodes:
      a = slicer.util.arrayFromVolume(self.nodes[nodeName])
      if len(a.shape) == 3: # skip the _u_rgb volume
        self.arrays[nodeName] = a
        self.means[nodeName] = a.mean()
        self.variances[nodeName] = a.var()
    self.names = [name for name in self.arrays]

  def indexAt(self,node,ras):
    """return the index for the node at the given RAS"""
    rasH = [1.,]*4
    rasH[:3] = ras
    rasToIJKMatrix = vtk.vtkMatrix4x4()
    node.GetRASToIJKMatrix(rasToIJKMatrix)
    ijkH = [0.,]*4
    rasToIJKMatrix.MultiplyPoint(rasH, ijkH)
    ijk = [int(round(element)) for element in ijkH[:3]]
    ijk.reverse()
    index = tuple(ijk)
    return(index)

  def sampleAtRAS(self,node,ras):
    """return the array sample value for the node at the given RAS"""
    index = self.indexAt(node,ras)
    return self.arrays[node.GetName()][index]

  def tableFromExperiment(self):
    """Makes a table of volume statistics and volume samples at fiducial point"""

    self.volumeStatistics();

    fiducials = slicer.util.getNode('ResearchMassSpecPoints')

    self.samples = {}
    for fiducialIndex in range(fiducials.GetNumberOfFiducials()):
      label = fiducials.GetNthFiducialLabel(fiducialIndex)
      self.samples[label] = {}
      ras = [0.,]*3
      fiducials.GetNthFiducialPosition(fiducialIndex, ras)
      for nodeName in self.names:
        node = self.nodes[nodeName]
        self.samples[label][nodeName] = self.sampleAtRAS(node, ras)
    self.labels = [label for label in self.samples]

    self.table = {}
    for i in range(0, len(self.labels)):
      for j in range(i+1, len(self.labels)):
        labelI, labelJ = self.labels[i], self.labels[j]
        datapointLabel = labelI + "_" + labelJ
        self.table[datapointLabel] = {}
        self.table[datapointLabel]['labelI'] = labelI
        self.table[datapointLabel]['labelJ'] = labelJ
        self.table[datapointLabel]['qti_distance'] = self.standardizedDistance([labelI, labelJ])
        self.table[datapointLabel]['intensity_difference'] = abs(self.intensities[labelI] - self.intensities[labelJ])
    return(self.table)

  def plotFromValues(self, title, axis_titles, datapoints):
    """Create a mrml Chart from given data"""

    print(title, axis_titles, datapoints)

    plotSeriesNodes = []
    for datapoint in datapoints:
      datapointLabel = datapoint[0]

      tableNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
      table = tableNode.GetTable()

      arrX = vtk.vtkFloatArray()
      arrX.SetName("labelIndex")
      table.AddColumn(arrX)

      arrY1 = vtk.vtkFloatArray()
      arrY1.SetName("qti_distance")
      table.AddColumn(arrY1)

      arrY2 = vtk.vtkFloatArray()
      arrY2.SetName("intensity_difference")
      table.AddColumn(arrY2)

      # Fill in the table with the values

      table.SetNumberOfRows(1)
      table.SetValue(0, 0, 0)
      table.SetValue(0, 1, datapoint[1])
      table.SetValue(0, 2, datapoint[2])

      # Create plot series node

      plotSeriesNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotSeriesNode", datapointLabel)
      plotSeriesNode.SetAndObserveTableNodeID(tableNode.GetID())
      plotSeriesNode.SetXColumnName("qti_distance")
      plotSeriesNode.SetYColumnName("intensity_difference")
      plotSeriesNode.SetPlotType(slicer.vtkMRMLPlotSeriesNode.PlotTypeScatter)
      plotSeriesNode.SetLineStyle(slicer.vtkMRMLPlotSeriesNode.LineStyleNone)
      plotSeriesNode.SetMarkerStyle(slicer.vtkMRMLPlotSeriesNode.MarkerStyleSquare)
      plotSeriesNode.SetUniqueColor()
      plotSeriesNodes.append(plotSeriesNode)

    # Create plot chart node

    plotChartNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLPlotChartNode")
    for plotSeriesNode in plotSeriesNodes:
      plotChartNode.AddAndObservePlotSeriesNodeID(plotSeriesNode.GetID())
    plotChartNode.SetTitle(title)
    plotChartNode.SetName(title)
    plotChartNode.SetXAxisTitle(axis_titles[0])
    plotChartNode.SetYAxisTitle(axis_titles[1])

    # Switch to a layout that contains a plot view to create a plot widget

    layoutManager = slicer.app.layoutManager()
    layoutWithPlot = slicer.modules.plots.logic().GetLayoutWithPlot(layoutManager.layout)
    layoutManager.setLayout(layoutWithPlot)

    # Select chart in plot view

    plotWidget = layoutManager.plotWidget(0)
    plotViewNode = plotWidget.mrmlPlotViewNode()
    plotViewNode.SetPlotChartNodeID(plotChartNode.GetID())

  def plotsFromTable(self):
    """Create a set of plots for different aspects of the table values"""
    contrastNames = list(self.nodes)
    scalarNames = list(filter(lambda n: not n.endswith('rgb'), contrastNames))

    for scalarName in scalarNames:
      datapoints = []
      for i in range(len(self.table.keys())):
        datapointLabel = list(self.table.keys())[i]
        self.table[datapointLabel]
        labelI = self.table[datapointLabel]['labelI']
        labelJ = self.table[datapointLabel]['labelJ']
        valueI = self.samples[labelI][scalarName]
        valueJ = self.samples[labelJ][scalarName]
        scalarDifference = abs(valueJ -valueI)
        datapoints.append([datapointLabel, scalarDifference, self.table[datapointLabel]['intensity_difference']])

      scalarLabel = scalarName[len('dtd_covariance_'):]
      self.plotFromValues(
        title = f'2HG Difference by {scalarLabel}',
        axis_titles = [f'Difference of {scalarLabel}', "Difference in 2HG m/z"],
        datapoints = datapoints)


  def mapFromCrosshair(self, targetNode=None):
    """Create a distance map based on the current location of the crosshair node"""

    self.volumeStatistics()

    if targetNode is None:
      targetNode = slicer.vtkSlicerVolumesLogic().CloneVolume(slicer.mrmlScene, self.nodes['dtd_covariance_MD'], 'distanceVolume')

    crosshairNode = slicer.util.getNode('vtkMRMLCrosshairNodedefault')
    crosshairRAS = crosshairNode.GetCrosshairRAS()
    crosshairIndex = self.indexAt(targetNode, crosshairRAS)

    print(f'RAS {crosshairRAS}')
    for nodeName in self.names:
      node = self.nodes[nodeName]
      sample = self.sampleAtRAS(node, crosshairRAS)
      print(f'{nodeName}: {sample}')

    targetArray = slicer.util.arrayFromVolume(targetNode)

    for element in numpy.ndenumerate(targetArray):
      index = element[0]
      distance = self.standardizedDistanceBetweenIndices(index, crosshairIndex)
      targetArray[index] = distance

    targetArray = slicer.util.updateVolumeFromArray(targetNode, targetArray)

  def segmentWithKMeans(self, targetSegmentationNode=None):
    """Use KMeans algorithm to segment using volumes"""

    self.volumeStatistics()

    if targetSegmentationNode is None:
      targetSegmentationNode = slicer.vtkSlicerVolumesLogic().CreateAndAddLabelVolume(slicer.mrmlScene, self.nodes['dtd_covariance_MD'], 'KMeans from QTI')
      targetSegmentationNode.CreateDefaultDisplayNodes()
      targetSegmentationNode.GetDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeLabels')
    targetArray = slicer.util.arrayFromVolume(targetSegmentationNode)

    trainingShape = (targetArray.flatten().shape[0], len(self.arrays.keys()))
    self.trainingArray = numpy.ndarray(trainingShape)

    index = 0
    for key in self.arrays.keys():
      self.trainingArray.T[index] = self.arrays[key].flatten()
      index += 1

    from sklearn.cluster import KMeans
    self.model = KMeans(n_clusters = 3).fit(self.trainingArray)

    print(self.model.labels_)

    targetArray[:] = self.model.labels_.reshape(targetArray.shape)

  def parallelCoordinatesPureD3(self):
    """Use parallel axes to explore parameter space

    See: http://syntagmatic.github.io/parallel-coordinates/

    Note also experimented with vtk version, but it has fewer features
    and performance is not any better in practice.

    https://vtk.org/Wiki/VTK/Examples/Python/Infovis/ParallelCoordinatesExtraction

    """

    self.volumeStatistics()

    fa = self.arrays['dtd_covariance_FA']
    indices = numpy.where(fa != 0)

    samples = {}
    for key in self.arrays.keys():
      samples[key] = self.arrays[key][indices]

    dataToPlot = []
    for index in range(len(samples['dtd_covariance_FA'])):
      indexData = {}
      for key in self.arrays.keys():
        scalarLabel = key[len('dtd_covariance_'):]
        indexData[scalarLabel] = samples[key][index]
      dataToPlot.append(indexData)

    dataToPlotString = json.dumps(dataToPlot)

    modulePath = os.path.dirname(slicer.modules.multimapper.path)
    resourceFilePath = os.path.join(modulePath, "Resources", "parallel-template.html")
    html = open(resourceFilePath).read().replace("%%dataToPlot%%", dataToPlotString)

    self.webWidget = slicer.qSlicerWebWidget()
    self.webWidget.size = qt.QSize(1024,512)
    self.webWidget.setHtml(html)
    self.webWidget.show()

    open('/tmp/data.html', 'w').write(html)

  def parallelCoordinatesSegmentation(self, segmentationNode=None, labelmapNode=None):
    """
    Like parallelCoordinatesParCoords_RandomSample below, but
    parcoords plot created from segmentation
    """

    self.volumeStatistics()

    if not segmentationNode:
      try:
        segmentationNode = slicer.util.getNode('Segmentation')
      except slicer.util.MRMLNodeNotFoundException:
        pass
    if not segmentationNode:
      print("need a segmentation")
      slicer.util.selectModule("SegmentEditor")
      return
    if not labelmapNode:
      try:
        labelmapNode = slicer.util.getNode('Segmentation-labelmap')
      except slicer.util.MRMLNodeNotFoundException:
        labelmapNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLLabelMapVolumeNode')
        labelmapNode.SetName("Segmentation-labelmap")
    segmentIDs = vtk.vtkStringArray()
    segmentationNode.GetSegmentation().GetSegmentIDs(segmentIDs)
    slicer.modules.segmentations.logic().ExportSegmentsToLabelmapNode(
                                            segmentationNode,
                                            segmentIDs,
                                            labelmapNode,
                                            self.nodes['dtd_covariance_FA'])
    labelArray = slicer.util.arrayFromVolume(labelmapNode)

    # list of non-zero segments in the labelmap
    segmentIndices = list(numpy.unique(labelArray))[1:]

    # data colors to match the segmentation colors to pass to js
    segmentation = segmentationNode.GetSegmentation()
    dataColors = [ [0,0,0] ]
    for colorNumber in range(len(segmentIndices)):
      dataColor = segmentation.GetNthSegment(colorNumber).GetColor()
      dataColors.append(dataColor)

    # make individual sample lines for parallel coordinates
    ijkToRAS = vtk.vtkMatrix4x4()
    labelmapNode.GetIJKToRASMatrix(ijkToRAS)
    rasCoordinates = []
    dataToPlot = []
    for segmentIndex in segmentIndices:
      # one data sample from each array per labeled voxel
      coordinates = numpy.transpose(numpy.where(labelArray == segmentIndex))
      self._coordinates = coordinates
      samples = {}
      for key in self.arrays.keys():
        samples[key] = []
        for coordinate in coordinates:
          self._array = self.arrays[key]
          samples[key].append(self.arrays[key][coordinate[0]][coordinate[1]][coordinate[2]])
      for coordinateNumber in range(len(coordinates)):
        indexData = {}
        indexData['segmentIndex'] = int(segmentIndex)
        for key in self.arrays.keys():
          scalarLabel = key[len('dtd_covariance_'):]
          indexData[scalarLabel] = samples[key][coordinateNumber]
        dataToPlot.append(indexData)
        ijkw = [*numpy.flip(coordinates)[coordinateNumber],1]
        ijkw = [*numpy.flip(coordinates)[coordinateNumber],1]
        rasCoordinate = ijkToRAS.MultiplyPoint(ijkw)
        rasCoordinates.append(rasCoordinate)

    dataColorsString = json.dumps(dataColors)
    dataToPlotString = json.dumps(dataToPlot)
    rasCoordinatesString = json.dumps(rasCoordinates)

    modulePath = os.path.dirname(slicer.modules.multimapper.path)
    resourceFilePath = os.path.join(modulePath, "Resources", "ParCoords-SEG-template.html")
    html = open(resourceFilePath).read()
    html = html.replace("%%dataColors%%", dataColorsString)
    html = html.replace("%%dataToPlot%%", dataToPlotString)
    html = html.replace("%%rasCoordinates%%", rasCoordinatesString)

    self.webWidget = slicer.qSlicerWebWidget()
    self.webWidget.size = qt.QSize(1024,768)
    self.webWidget.setHtml(html)
    self.webWidget.show()

    def crosshairCallback(observer,eventID):
      crosshairNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode')
      ras = [0,]*3
      crosshairNode.GetCursorPositionRAS(ras)
      print(ras)
      # TODO: update selector ranges based on QTI statistics

    crosshairNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode')
    event = vtk.vtkCommand.ModifiedEvent
    id_ = crosshairNode.AddObserver(event, crosshairCallback)
    self.observerObjectIDPairs.append((crosshairNode, id_))

    # save for debugging
    open('/tmp/data.html', 'w').write(html)

  def parallelCoordinatesParCoords_RandomSample(self,sampleSize=1000):
    """Use parallel axes to explore parameter space

    See: http://syntagmatic.github.io/parallel-coordinates/
    https://github.com/BigFatDog/parcoords-es

    Note also experimented with vtk version, but it has fewer features
    and performance is not any better in practice.

    https://vtk.org/Wiki/VTK/Examples/Python/Infovis/ParallelCoordinatesExtraction

    """

    self.volumeStatistics()

    fa = self.arrays['dtd_covariance_FA']
    indices = numpy.where(fa != 0)

    ijkCoordinates = numpy.transpose(indices)
    ijkToRAS = vtk.vtkMatrix4x4()
    slicer.util.getNode('dtd_covariance_FA').GetIJKToRASMatrix(ijkToRAS)
    rasCoordinates = []

    samples = {}
    for key in self.arrays.keys():
      samples[key] = self.arrays[key][indices]

    dataToPlot = []
    randomSample = random.sample(range(len(samples['dtd_covariance_FA'])), sampleSize)
    sampleIndex = 0
    for index in randomSample:
      indexData = {}
      for key in self.arrays.keys():
        scalarLabel = key[len('dtd_covariance_'):]
        indexData[scalarLabel] = samples[key][index]
      dataToPlot.append(indexData)
      ijk = [*numpy.flip(numpy.transpose(indices)[index]),1]
      rasCoordinates.append(ijkToRAS.MultiplyPoint(ijk))
      sampleIndex += 1

    dataToPlotString = json.dumps(dataToPlot)
    rasCoordinatesString = json.dumps(rasCoordinates)

    modulePath = os.path.dirname(slicer.modules.multimapper.path)
    resourceFilePath = os.path.join(modulePath, "Resources", "ParCoords-template.html")
    html = open(resourceFilePath).read()
    html = html.replace("%%dataToPlot%%", dataToPlotString)
    html = html.replace("%%rasCoordinates%%", rasCoordinatesString)

    self.webWidget = slicer.qSlicerWebWidget()
    self.webWidget.size = qt.QSize(1024,768)
    self.webWidget.setHtml(html)
    self.webWidget.show()

    def crosshairCallback(observer,eventID):
      crosshairNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode')
      ras = [0,]*3
      crosshairNode.GetCursorPositionRAS(ras)
      print(ras)
      # TODO: update selector ranges based on QTI statistics

    crosshairNode = slicer.mrmlScene.GetFirstNodeByClass('vtkMRMLCrosshairNode')
    event = vtk.vtkCommand.ModifiedEvent
    id_ = crosshairNode.AddObserver(event, crosshairCallback)
    self.observerObjectIDPairs.append((crosshairNode, id_))

    # save for debugging
    open('/tmp/data.html', 'w').write(html)

  def segmentFromExtents(self, ranges):
    """Use ranges to segment using volumes
       Extents will typically come from ParCoords callback
       (the call actually comes from Resources/ParCoordsSegmentation-template.html)
    """

    self.volumeStatistics()

    #
    # create targetArray to point at labelmap
    # (create labemap if needed to match dtd_covariance_MD)
    #
    segmentationName = 'Label from QTI Extents'
    try:
      targetLabelmapNode = slicer.util.getNode(segmentationName+'*')
    except slicer.util.MRMLNodeNotFoundException:
      targetLabelmapNode = slicer.vtkSlicerVolumesLogic().CreateAndAddLabelVolume(slicer.mrmlScene, self.nodes['dtd_covariance_MD'], segmentationName)
      targetLabelmapNode.CreateDefaultDisplayNodes()
      targetLabelmapNode.GetDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeLabels')
      slicer.util.setSliceViewerLayers(label=targetLabelmapNode.GetID())
    targetArray = slicer.util.arrayFromVolume(targetLabelmapNode)
    targetArray = numpy.ones_like(targetArray)

    #
    # make a key mask which is 1 for each image
    # where the QTI signal is in the range
    # and the FA is non-zero, then update
    # the segmentation
    #
    for rangeKey in ranges.keys():
      key = "dtd_covariance_" + rangeKey
      if not key in self.arrays:
        continue
      keyArray = self.arrays["dtd_covariance_" + rangeKey]
      keyMask = numpy.zeros_like(keyArray)
      keyExtents = ranges[rangeKey]
      if len(keyExtents) > 0:
        for keyRange in keyExtents:
          upper,lower = keyRange['selection']['scaled']
          indices = numpy.where(numpy.logical_and(keyArray >= lower, keyArray <= upper))
          keyMask[indices] = 1
        targetArray = targetArray * keyMask
    brainMask = numpy.clip(numpy.ceil(self.arrays["dtd_covariance_FA"]), 0, 1)
    targetArray = targetArray * brainMask
    slicer.targetArray = targetArray
    slicer.util.updateVolumeFromArray(targetLabelmapNode, targetArray)


class MultiMapperTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear(0)

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_MultiMapper1()

  def test_MultiMapper1(self):
    """
    Placeholder test
    """

    self.delayDisplay("Starting the test")

    self.delayDisplay('Test passed!')
