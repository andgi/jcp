package se.hb.jcp.cli;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.List;

public class AppView extends JFrame {

    private JButton fileButton;
    private JButton deleteButton; 
    private JList<String> fileList;
    private DefaultListModel<String> listModel;
    private JList<String> testList; 
    private DefaultListModel<String> testListModel; 
    private AppPresenter presenter;
    private JButton predictButton;
    private JSpinner spinner;

    private JButton testFileButton;

    public AppView() {
        super("Conformal Regression Application");

        fileButton = new JButton("Select Set");
        testFileButton = new JButton("Select Test Set");
        deleteButton = new JButton("Delete Selected File"); 
        listModel = new DefaultListModel<>();
        fileList = new JList<>(listModel);
        testListModel = new DefaultListModel<>(); 
        testList = new JList<>(testListModel); 
        predictButton = new JButton("Predict");
        spinner = new JSpinner(new SpinnerNumberModel(0.0, 0.0, 1.0, 0.01));
        spinner.setValue(0.1);

        presenter = new AppPresenter(this);

        fileButton.addActionListener(e -> presenter.selectSets());

        predictButton.addActionListener(e -> presenter.launchPrediction(spinner.getValue().toString()));

        deleteButton.addActionListener(new ActionListener() {
            @Override
            public void actionPerformed(ActionEvent e) {
                int selectedIndex = fileList.getSelectedIndex();
                if (selectedIndex != -1) {
                    listModel.remove(selectedIndex);
                }
            }
        });

        getContentPane().setLayout(new BorderLayout()); 
        JPanel westPanel = new JPanel();
        westPanel.setLayout(new BoxLayout(westPanel, BoxLayout.Y_AXIS)); 

        JPanel inputPanel = new JPanel(); 
        inputPanel.setLayout(new BoxLayout(inputPanel, BoxLayout.LINE_AXIS)); 
        int padding = 10; 
        inputPanel.setBorder(BorderFactory.createEmptyBorder(padding, padding, padding, padding)); 

        inputPanel.add(Box.createHorizontalStrut(padding)); 
        inputPanel.add(fileButton);
        inputPanel.add(Box.createHorizontalStrut(padding));
        inputPanel.add(new JLabel("Significance Level"));
        inputPanel.add(Box.createHorizontalStrut(padding)); 
        inputPanel.add(spinner);

        westPanel.add(inputPanel);
        westPanel.add(new JScrollPane(fileList));
        add(westPanel, BorderLayout.WEST); 

    
        JPanel eastPanel = new JPanel();
        eastPanel.setLayout(new BoxLayout(eastPanel, BoxLayout.Y_AXIS)); 

        JPanel inputTestPanel = new JPanel(); 
        inputTestPanel.setLayout(new BoxLayout(inputTestPanel, BoxLayout.LINE_AXIS)); 
        inputTestPanel.setBorder(BorderFactory.createEmptyBorder(padding, padding, padding, padding)); 

        inputTestPanel.add(Box.createHorizontalStrut(padding)); 
        inputTestPanel.add(testFileButton);

        testFileButton.addActionListener(e -> presenter.selectTestSets());

        eastPanel.add(inputTestPanel);
        eastPanel.add(new JScrollPane(testList)); 
        add(eastPanel, BorderLayout.EAST);

        
        add(deleteButton, BorderLayout.SOUTH); 
        add(predictButton, BorderLayout.SOUTH); 

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(600, 400);
        setLocationRelativeTo(null);
        setVisible(true);
    }

    public void addFileName(String fileName) {
        listModel.addElement(fileName);
    }
    public void addFileNameTest(String fileName) {
        testListModel.addElement(fileName);
    }

    public List<String> getSelectedFiles() {
        return fileList.getSelectedValuesList();
    }
}
