clc;
clear all;
close all;

FILE_PATH = 'C:\Users\osama\OneDrive - Georgia Institute of Technology\Frequency Extrapolation\18_third_application\Data\';

file_S11 = open([FILE_PATH, 'CPW_A2.S2P']);
file_S21 = open([FILE_PATH, 'CPW_A6.S2P']);
file_S12 = open([FILE_PATH, 'CPW_b2.S2P']);
file_S22 = open([FILE_PATH, 'CPW_b6.S2P']);
