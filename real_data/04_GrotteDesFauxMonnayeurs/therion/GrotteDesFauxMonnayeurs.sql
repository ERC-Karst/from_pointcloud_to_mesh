create table SURVEY (ID integer, PARENT_ID integer, NAME varchar(23), FULL_NAME varchar(23), TITLE varchar(28));
create table CENTRELINE (ID integer, SURVEY_ID integer, TITLE varchar(4), TOPO_DATE date, EXPLO_DATE date, LENGTH real, SURFACE_LENGTH real, DUPLICATE_LENGTH real);
create table PERSON (ID integer, NAME varchar(11), SURNAME varchar(15));
create table EXPLO (PERSON_ID integer, CENTRELINE_ID integer);
create table TOPO (PERSON_ID integer, CENTRELINE_ID integer);
create table STATION (ID integer, NAME varchar(4), SURVEY_ID integer, X real, Y real, Z real);
create table STATION_FLAG (STATION_ID integer, FLAG char(3));
create table SHOT (ID integer, FROM_ID integer, TO_ID integer, CENTRELINE_ID integer, LENGTH real, BEARING real, GRADIENT real, ADJ_LENGTH real, ADJ_BEARING real, ADJ_GRADIENT real, ERR_LENGTH real, ERR_BEARING real, ERR_GRADIENT real);
create table SHOT_FLAG (SHOT_ID integer, FLAG char(3));
create table MAPS (ID integer, SURVEY_ID integer, NAME varchar(23), TITLE varchar(28), PROJID integer, LENGTH real, DEPTH real);
create table SCRAPS (ID integer, SURVEY_ID integer, NAME varchar(23), PROJID integer, MAX_DISTORTION real, AVG_DISTORTION real);
create table MAPITEMS (ID integer, TYPE integer, ITEMID integer);
insert into SURVEY values (1, 0, '', '', NULL);
 insert into CENTRELINE values (2, 1, NULL, NULL, NULL, 0.00, 0.00, 0.00);
 insert into SURVEY values (19, 1, 'GrotteDesFauxMonnayeurs', 'GrotteDesFauxMonnayeurs', 'Grotte des Faux Monnayeurs');
 insert into CENTRELINE values (20, 19, NULL, NULL, NULL, 0.00, 0.00, 0.00);
 insert into CENTRELINE values (21, 19, NULL, '2023-09-01', NULL, 447.47, 0.00, 0.00);
 insert into PERSON values (1, 'Ana Paula', 'Burgoa Tanaka');
insert into TOPO values (1, 21);
 insert into PERSON values (2, 'Jeremie', 'Chappuis');
insert into TOPO values (2, 21);
 insert into PERSON values (3, 'Tanguy', 'Racine');
insert into TOPO values (3, 21);
 insert into PERSON values (4, 'Philippe', 'Renard');
insert into TOPO values (4, 21);
 insert into PERSON values (5, 'Robin', 'Voland');
insert into TOPO values (5, 21);
 insert into SHOT values (1, 1, 2, 21, 2.170, 132.62, 9.30, 2.173, 132.54, 9.27, 0.004, 89.45, -9.31);
insert into SHOT values (2, 1, 3, 21, 11.130, 226.92, -8.50, 11.132, 226.91, -8.52, 0.005, 182.17, -74.26);
insert into SHOT values (3, 1, 4, 21, 10.840, 226.92, -8.60, 10.862, 229.98, -8.36, 0.576, 315.65, 4.08);
insert into SHOT values (4, 4, 1, 21, 10.870, 53.02, 8.10, 10.862, 49.98, 8.36, 0.572, 319.96, 4.85);
insert into SHOT values (5, 4, 5, 21, 8.940, 317.02, -17.10, 8.943, 314.53, -16.90, 0.373, 227.54, 4.41);
insert into SHOT values (6, 5, 4, 21, 8.950, 131.92, 16.70, 8.943, 134.53, 16.90, 0.391, 225.61, 4.12);
insert into SHOT values (7, 5, 6, 21, 5.810, 266.22, -13.30, 5.777, 266.84, -13.01, 0.076, 19.18, 28.81);
insert into SHOT values (8, 6, 5, 21, 5.730, 87.52, 12.70, 5.777, 86.84, 13.01, 0.087, 27.51, 27.71);
insert into SHOT values (9, 6, 7, 21, 3.790, 343.02, -16.00, 3.759, 339.11, -15.75, 0.250, 245.22, 5.66);
insert into SHOT values (10, 7, 6, 21, 3.780, 155.02, 15.40, 3.759, 159.11, 15.75, 0.261, 252.89, 3.56);
insert into SHOT values (11, 7, 8, 21, 5.530, 237.92, 1.90, 5.525, 240.37, 1.97, 0.237, 330.34, 1.61);
insert into SHOT values (12, 8, 7, 21, 5.520, 62.82, -2.10, 5.525, 60.37, -1.97, 0.236, 332.99, 2.98);
insert into SHOT values (13, 8, 9, 21, 3.110, 286.52, -19.20, 3.105, 286.64, -19.18, 0.008, 49.71, 20.10);
insert into SHOT values (14, 8, 10, 21, 9.020, 252.22, -4.30, 9.019, 253.40, -4.20, 0.186, 342.70, 5.03);
insert into SHOT values (15, 10, 8, 21, 9.020, 74.52, 4.10, 9.019, 73.40, 4.20, 0.177, 343.32, 4.90);
insert into SHOT values (16, 10, 11, 21, 5.710, 224.02, 6.30, 5.726, 227.14, 6.52, 0.310, 313.06, 4.33);
insert into SHOT values (17, 11, 10, 21, 5.750, 50.32, -6.70, 5.726, 47.14, -6.52, 0.318, 314.82, 3.76);
insert into SHOT values (18, 11, 12, 21, 10.770, 270.42, 8.10, 10.782, 270.54, 8.27, 0.040, 340.65, 54.33);
insert into SHOT values (19, 12, 11, 21, 10.810, 90.62, -8.40, 10.782, 90.54, -8.27, 0.040, 303.59, 46.05);
insert into SHOT values (20, 12, 13, 21, 8.350, 237.52, 7.60, 8.370, 239.57, 7.83, 0.299, 325.52, 6.85);
insert into SHOT values (21, 13, 12, 21, 8.380, 61.62, -8.10, 8.370, 59.57, -7.83, 0.300, 329.81, 7.82);
insert into SHOT values (22, 13, 14, 21, 9.490, 259.42, 14.70, 9.491, 259.90, 14.65, 0.078, 347.53, -6.02);
insert into SHOT values (23, 14, 13, 21, 9.500, 80.32, -14.60, 9.491, 79.90, -14.65, 0.068, 340.79, -4.50);
insert into SHOT values (24, 14, 15, 21, 2.260, 259.72, -21.40, 2.259, 259.60, -21.29, 0.006, 174.84, 45.92);
insert into SHOT values (25, 14, 16, 21, 15.570, 242.12, 1.10, 15.553, 243.78, 1.14, 0.452, 335.14, 1.41);
insert into SHOT values (26, 16, 14, 21, 15.520, 65.42, -1.20, 15.553, 63.78, -1.14, 0.446, 338.89, 1.93);
insert into SHOT values (27, 16, 17, 21, 9.280, 212.22, -6.10, 9.282, 215.88, -6.00, 0.589, 303.67, 1.57);
insert into SHOT values (28, 17, 16, 21, 9.290, 39.62, 5.90, 9.282, 35.88, 6.00, 0.603, 306.86, 1.43);
insert into SHOT values (29, 17, 18, 21, 11.910, 242.32, -15.00, 11.937, 244.19, -15.00, 0.377, 329.30, -1.13);
insert into SHOT values (30, 18, 17, 21, 11.950, 66.12, 15.00, 11.937, 64.19, 15.00, 0.389, 333.29, -0.43);
insert into SHOT values (31, 18, 19, 21, 7.100, 65.32, -3.60, 7.101, 65.32, -3.55, 0.006, 49.38, 73.18);
insert into SHOT values (32, 18, 20, 21, 18.840, 217.62, 0.50, 18.879, 220.90, 0.55, 1.080, 307.17, 0.83);
insert into SHOT values (33, 20, 18, 21, 18.880, 44.12, -0.60, 18.879, 40.90, -0.55, 1.061, 312.49, 0.96);
insert into SHOT values (34, 20, 21, 21, 19.630, 233.42, -5.80, 19.620, 235.69, -5.91, 0.775, 325.59, -2.68);
insert into SHOT values (35, 21, 20, 21, 19.580, 58.02, 6.00, 19.620, 55.69, 5.91, 0.794, 329.94, -1.93);
insert into SHOT values (36, 21, 22, 21, 15.320, 219.32, -5.50, 15.372, 222.46, -5.45, 0.838, 307.25, 0.57);
insert into SHOT values (37, 22, 21, 21, 15.400, 45.52, 5.40, 15.372, 42.46, 5.45, 0.819, 311.96, 0.75);
insert into SHOT values (38, 22, 23, 21, 5.010, 218.92, 10.60, 5.013, 218.99, 10.58, 0.007, 282.78, -13.85);
insert into SHOT values (39, 22, 24, 21, 23.150, 236.32, 2.10, 23.170, 238.58, 2.10, 0.914, 326.18, 0.11);
insert into SHOT values (40, 24, 22, 21, 23.160, 60.82, -2.10, 23.170, 58.58, -2.10, 0.904, 330.35, -0.08);
insert into SHOT values (41, 24, 25, 21, 7.710, 273.72, 0.30, 7.706, 273.72, 0.30, 0.004, 91.59, -5.76);
insert into SHOT values (42, 24, 26, 21, 5.880, 1.62, 3.30, 5.882, 1.66, 3.31, 0.005, 63.16, 18.19);
insert into SHOT values (43, 24, 27, 21, 8.200, 267.02, -3.90, 8.209, 267.13, -3.91, 0.019, 326.91, -7.04);
insert into SHOT values (44, 27, 24, 21, 8.220, 87.32, 3.90, 8.209, 87.13, 3.91, 0.029, 335.59, 1.81);
insert into SHOT values (45, 27, 28, 21, 6.870, 209.72, -7.50, 6.878, 213.76, -7.43, 0.481, 300.69, 0.80);
insert into SHOT values (46, 28, 27, 21, 6.900, 37.72, 7.40, 6.878, 33.76, 7.43, 0.472, 303.01, 0.16);
insert into SHOT values (47, 28, 29, 21, 0.990, 147.02, 0.70, 0.990, 146.95, 0.58, 0.002, 70.35, -60.32);
insert into SHOT values (48, 28, 30, 21, 1.020, 146.62, 6.30, 1.016, 146.31, 6.22, 0.007, 18.33, -15.53);
insert into SHOT values (49, 28, 31, 21, 0.990, 146.12, 15.50, 0.992, 145.65, 15.20, 0.010, 76.23, -28.47);
insert into SHOT values (50, 28, 32, 21, 1.420, 150.62, 6.00, 1.423, 150.36, 6.05, 0.007, 85.38, 12.35);
insert into SHOT values (51, 28, 33, 21, 1.580, 148.82, 29.60, 1.580, 148.39, 29.59, 0.010, 58.54, -2.40);
insert into SHOT values (52, 28, 34, 21, 2.090, 150.12, 45.40, 2.093, 149.77, 45.39, 0.009, 74.81, 11.44);
insert into SHOT values (53, 28, 35, 21, 3.260, 151.92, 57.60, 3.259, 151.68, 57.54, 0.008, 81.42, -17.75);
insert into SHOT values (54, 28, 36, 21, 3.760, 160.12, 70.60, 3.762, 159.82, 70.65, 0.008, 49.63, 26.67);
insert into SHOT values (55, 28, 37, 21, 4.030, 175.72, 78.40, 4.033, 175.76, 78.38, 0.003, 194.32, 49.45);
insert into SHOT values (56, 28, 38, 21, 4.230, 203.22, 83.60, 4.226, 202.71, 83.67, 0.008, 60.58, -28.27);
insert into SHOT values (57, 28, 39, 21, 4.810, 285.92, 83.50, 4.811, 286.09, 83.54, 0.004, 79.80, 14.01);
insert into SHOT values (58, 28, 40, 21, 5.170, 296.72, 79.10, 5.173, 296.83, 79.14, 0.005, 82.18, 45.08);
insert into SHOT values (59, 28, 41, 21, 4.950, 351.62, 82.40, 4.954, 352.12, 82.39, 0.007, 66.81, 30.57);
insert into SHOT values (60, 28, 42, 21, 5.780, 338.12, 75.20, 5.782, 338.13, 75.21, 0.002, 114.02, 81.13);
insert into SHOT values (61, 28, 43, 21, 6.210, 330.92, 66.40, 6.207, 331.06, 66.45, 0.009, 108.14, -3.97);
insert into SHOT values (62, 28, 44, 21, 5.830, 328.32, 59.20, 5.830, 328.44, 59.25, 0.008, 92.74, 16.17);
insert into SHOT values (63, 28, 45, 21, 6.040, 326.92, 48.00, 6.041, 327.02, 48.01, 0.007, 59.02, 11.41);
insert into SHOT values (64, 28, 46, 21, 6.070, 329.22, 37.20, 6.073, 329.30, 37.18, 0.008, 34.99, 0.64);
insert into SHOT values (65, 28, 47, 21, 6.240, 329.42, 25.50, 6.240, 329.47, 25.54, 0.007, 77.09, 33.23);
insert into SHOT values (66, 28, 48, 21, 4.170, 346.22, 15.40, 4.174, 346.34, 15.42, 0.010, 54.60, 15.62);
insert into SHOT values (67, 28, 49, 21, 5.680, 339.42, 16.10, 5.679, 339.51, 16.15, 0.010, 82.71, 29.88);
insert into SHOT values (68, 28, 50, 21, 6.100, 340.72, 25.80, 6.102, 340.77, 25.74, 0.009, 29.44, -35.06);
insert into SHOT values (69, 28, 51, 21, 6.080, 343.42, 40.80, 6.077, 343.46, 40.79, 0.005, 108.54, -37.80);
insert into STATION values (1, '0', 19, 900393.34, 2233088.83, 451.88);
insert into STATION values (2, 'c8', 19, 900394.92, 2233087.38, 452.23);
insert into STATION values (3, '.', 19, 900385.30, 2233081.31, 450.23);
insert into STATION values (4, '1', 19, 900385.11, 2233081.92, 450.30);
insert into STATION values (5, '2', 19, 900379.01, 2233087.92, 447.70);
insert into STATION values (6, '3', 19, 900373.39, 2233087.61, 446.40);
insert into STATION values (7, '4', 19, 900372.10, 2233090.99, 445.38);
insert into STATION values (8, '5', 19, 900367.30, 2233088.26, 445.57);
insert into STATION values (9, 'c7', 19, 900364.49, 2233089.10, 444.55);
insert into STATION values (10, '6', 19, 900358.68, 2233085.69, 444.91);
insert into STATION values (11, '7', 19, 900354.51, 2233081.82, 445.56);
insert into STATION values (12, '8', 19, 900343.84, 2233081.92, 447.11);
insert into STATION values (13, '9', 19, 900336.69, 2233077.72, 448.25);
insert into STATION values (14, '10', 19, 900327.65, 2233076.11, 450.65);
insert into STATION values (15, 'c6', 19, 900325.58, 2233075.73, 449.83);
insert into STATION values (16, '11', 19, 900313.70, 2233069.24, 450.96);
insert into STATION values (17, '12', 19, 900308.29, 2233061.76, 449.99);
insert into STATION values (18, '13', 19, 900297.91, 2233056.74, 446.90);
insert into STATION values (19, 'c5', 19, 900304.35, 2233059.70, 446.46);
insert into STATION values (20, '14', 19, 900285.55, 2233042.47, 447.08);
insert into STATION values (21, '15', 19, 900269.43, 2233031.47, 445.06);
insert into STATION values (22, '16', 19, 900259.10, 2233020.18, 443.60);
insert into STATION values (23, 'c2', 19, 900256.00, 2233016.35, 444.52);
insert into STATION values (24, '17', 19, 900239.34, 2233008.11, 444.45);
insert into STATION values (25, 'c1', 19, 900231.65, 2233008.61, 444.49);
insert into STATION values (26, 'c3', 19, 900239.51, 2233013.98, 444.79);
insert into STATION values (27, '18', 19, 900231.16, 2233007.70, 443.89);
insert into STATION values (28, '19', 19, 900227.37, 2233002.03, 443.00);
insert into STATION_FLAG values(28, 'fix');
insert into STATION values (29, '.', 19, 900227.91, 2233001.20, 443.01);
insert into STATION values (30, '.', 19, 900227.93, 2233001.19, 443.11);
insert into STATION values (31, '.', 19, 900227.91, 2233001.24, 443.26);
insert into STATION values (32, '.', 19, 900228.07, 2233000.80, 443.15);
insert into STATION values (33, '.', 19, 900228.09, 2233000.86, 443.78);
insert into STATION values (34, '.', 19, 900228.11, 2233000.76, 444.49);
insert into STATION values (35, '.', 19, 900228.20, 2233000.49, 445.75);
insert into STATION values (36, '.', 19, 900227.80, 2233000.86, 446.55);
insert into STATION values (37, '.', 19, 900227.43, 2233001.22, 446.95);
insert into STATION values (38, '.', 19, 900227.19, 2233001.60, 447.20);
insert into STATION values (39, '.', 19, 900226.85, 2233002.18, 447.78);
insert into STATION values (40, '.', 19, 900226.50, 2233002.47, 448.08);
insert into STATION values (41, '.', 19, 900227.28, 2233002.68, 447.91);
insert into STATION values (42, '.', 19, 900226.82, 2233003.40, 448.59);
insert into STATION values (43, '.', 19, 900226.17, 2233004.20, 448.69);
insert into STATION values (44, '.', 19, 900225.81, 2233004.57, 448.01);
insert into STATION values (45, '.', 19, 900225.17, 2233005.42, 447.49);
insert into STATION values (46, '.', 19, 900224.90, 2233006.19, 446.67);
insert into STATION values (47, '.', 19, 900224.51, 2233006.88, 445.69);
insert into STATION values (48, '.', 19, 900226.42, 2233005.94, 444.11);
insert into STATION values (49, '.', 19, 900225.46, 2233007.14, 444.58);
insert into STATION values (50, '.', 19, 900225.56, 2233007.22, 445.65);
insert into STATION values (51, '.', 19, 900226.06, 2233006.44, 446.97);
