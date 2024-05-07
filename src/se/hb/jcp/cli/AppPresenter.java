package se.hb.jcp.cli;

import javax.swing.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.io.IOException;

public class AppPresenter {

    private AppView _view;
    private List<File> _selectedSets;
    private List<File> _selectedTestSets;
    public AppPresenter(AppView view) {
        _view = view;
        _selectedSets = new ArrayList<>();
        _selectedTestSets = new ArrayList<>();
    }

    public void selectSets() {
        JFileChooser fileChooser = new JFileChooser("../pisvm-datasets/classification");

        int returnValue = fileChooser.showOpenDialog(null);

        if (returnValue == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            _selectedSets.add(selectedFile);
            _view.addFileName(selectedFile.getName()); 
        }
    }

    public void selectTestSets() {
        JFileChooser fileChooser = new JFileChooser("../pisvm-datasets/classification");

        int returnValue = fileChooser.showOpenDialog(null);

        if (returnValue == JFileChooser.APPROVE_OPTION) {
            File selectedFile = fileChooser.getSelectedFile();
            _selectedTestSets.add(selectedFile);
            _view.addFileNameTest(selectedFile.getName()); 
        }
    }

    public void launchPrediction(String significance) {
     
        if(_selectedSets.size() == _selectedTestSets.size()) {
            for (int i = 0; i < _selectedSets.size(); i ++) {
                jcp_train model = new jcp_train();
                jcp_predict predictions = new jcp_predict();

                String fileName = _selectedSets.get(i).getName();
                int lastDotIndex = fileName.lastIndexOf(".");
                String fileNameModel = fileName.substring(0, lastDotIndex) + ".model";
                // FIXME ADD DIFFERENT ARGUMENTS USER SELECTABLE  
                String[] argsModel = {"-c", "1", "-s", significance, "-m", fileNameModel, _selectedSets.get(i).getAbsolutePath()};
        
                try {       
                    model.run(argsModel);
                    
                } catch (IOException e) {
                    e.printStackTrace();
                }

                String[] argsPredict = {"-m", fileNameModel, _selectedTestSets.get(i).getAbsolutePath()};
                try {
                    predictions.run(argsPredict);
                } catch(IOException e) {
                    e.printStackTrace();
                }
            }
        }
            
        
    }
}
