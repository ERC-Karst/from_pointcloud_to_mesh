create table SURVEY (ID integer, PARENT_ID integer, NAME varchar(11), FULL_NAME varchar(11), TITLE varchar(21));
create table CENTRELINE (ID integer, SURVEY_ID integer, TITLE varchar(4), TOPO_DATE date, EXPLO_DATE date, LENGTH real, SURFACE_LENGTH real, DUPLICATE_LENGTH real);
create table PERSON (ID integer, NAME varchar(11), SURNAME varchar(15));
create table EXPLO (PERSON_ID integer, CENTRELINE_ID integer);
create table TOPO (PERSON_ID integer, CENTRELINE_ID integer);
create table STATION (ID integer, NAME varchar(4), SURVEY_ID integer, X real, Y real, Z real);
create table STATION_FLAG (STATION_ID integer, FLAG char(3));
create table SHOT (ID integer, FROM_ID integer, TO_ID integer, CENTRELINE_ID integer, LENGTH real, BEARING real, GRADIENT real, ADJ_LENGTH real, ADJ_BEARING real, ADJ_GRADIENT real, ERR_LENGTH real, ERR_BEARING real, ERR_GRADIENT real);
create table SHOT_FLAG (SHOT_ID integer, FLAG char(3));
create table MAPS (ID integer, SURVEY_ID integer, NAME varchar(11), TITLE varchar(21), PROJID integer, LENGTH real, DEPTH real);
create table SCRAPS (ID integer, SURVEY_ID integer, NAME varchar(11), PROJID integer, MAX_DISTORTION real, AVG_DISTORTION real);
create table MAPITEMS (ID integer, TYPE integer, ITEMID integer);
insert into SURVEY values (1, 0, '', '', NULL);
 insert into CENTRELINE values (2, 1, NULL, NULL, NULL, 0.00, 0.00, 0.00);
 insert into SURVEY values (19, 1, 'LesCavottes', 'LesCavottes', 'Grotte des Cavottes');
 insert into CENTRELINE values (20, 19, NULL, NULL, NULL, 0.00, 0.00, 0.00);
 insert into CENTRELINE values (21, 19, NULL, '2023-08-31', NULL, 319.73, 0.00, 256.41);
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
 insert into SHOT values (1, 1, 2, 21, 6.830, 48.23, 2.20, 6.826, 48.27, 2.27, 0.010, 182.78, 51.03);
insert into SHOT values (2, 1, 3, 21, 7.690, 142.03, -15.30, 7.734, 144.33, -15.29, 0.301, 225.13, -2.06);
insert into SHOT values (3, 3, 1, 21, 7.780, 326.63, 15.30, 7.734, 324.33, 15.29, 0.304, 227.04, -2.44);
insert into SHOT_FLAG values(3, 'dpl');
insert into SHOT values (4, 3, 4, 21, 7.050, 192.83, -11.10, 7.033, 196.68, -11.15, 0.464, 286.97, -0.34);
insert into SHOT values (5, 4, 3, 21, 7.030, 20.63, 11.20, 7.033, 16.68, 11.15, 0.476, 289.14, -0.66);
insert into SHOT_FLAG values(5, 'dpl');
insert into SHOT values (6, 4, 5, 21, 2.890, 207.93, 1.90, 2.832, 210.82, 2.02, 0.156, 321.49, 1.54);
insert into SHOT values (7, 5, 4, 21, 2.790, 33.73, -2.30, 2.832, 30.82, -2.02, 0.149, 318.69, 4.61);
insert into SHOT_FLAG values(7, 'dpl');
insert into SHOT values (8, 5, 6, 21, 14.280, 147.73, -2.00, 14.295, 151.89, -2.20, 1.038, 239.07, -2.85);
insert into SHOT values (9, 6, 5, 21, 14.280, 336.03, 2.40, 14.295, 331.89, 2.20, 1.032, 244.92, -2.66);
insert into SHOT_FLAG values(9, 'dpl');
insert into SHOT values (10, 6, 7, 21, 12.380, 129.43, 21.60, 12.387, 132.04, 21.65, 0.525, 220.49, 1.38);
insert into SHOT values (11, 7, 6, 21, 12.380, 314.63, -21.60, 12.387, 312.04, -21.65, 0.520, 223.59, -1.39);
insert into SHOT_FLAG values(11, 'dpl');
insert into SHOT values (12, 7, 8, 21, 6.770, 290.43, -17.50, 6.773, 290.48, -17.53, 0.007, 4.65, -34.52);
insert into SHOT values (13, 7, 9, 21, 4.640, 259.13, -30.10, 4.649, 259.42, -29.94, 0.026, 313.75, 15.96);
insert into SHOT values (14, 9, 7, 21, 4.670, 79.83, 29.70, 4.649, 79.42, 29.94, 0.041, 305.99, 8.71);
insert into SHOT_FLAG values(14, 'dpl');
insert into SHOT values (15, 9, 10, 21, 10.960, 154.63, 11.80, 10.953, 158.63, 11.64, 0.748, 246.66, -2.40);
insert into SHOT values (16, 10, 9, 21, 10.950, 342.63, -11.40, 10.953, 338.63, -11.64, 0.752, 250.17, -3.48);
insert into SHOT_FLAG values(16, 'dpl');
insert into SHOT values (17, 10, 11, 21, 15.130, 95.83, 11.20, 15.145, 96.69, 11.12, 0.223, 181.30, -4.83);
insert into SHOT values (18, 11, 10, 21, 15.150, 277.53, -11.10, 15.145, 276.69, -11.12, 0.220, 185.66, -0.86);
insert into SHOT_FLAG values(18, 'dpl');
insert into SHOT values (19, 11, 12, 21, 11.110, 214.43, -11.20, 11.149, 217.27, -11.33, 0.542, 302.27, -3.39);
insert into SHOT values (20, 12, 11, 21, 11.170, 40.03, 11.50, 11.149, 37.27, 11.33, 0.529, 307.18, -4.01);
insert into SHOT_FLAG values(20, 'dpl');
insert into SHOT values (21, 12, 13, 21, 4.720, 193.13, 16.20, 4.710, 196.71, 16.28, 0.283, 287.28, 0.64);
insert into SHOT values (22, 13, 12, 21, 4.730, 20.43, -16.40, 4.710, 16.71, -16.28, 0.295, 285.34, 3.01);
insert into SHOT_FLAG values(22, 'dpl');
insert into SHOT values (23, 13, 14, 21, 7.300, 251.93, -2.50, 7.292, 252.58, -2.67, 0.086, 348.55, -14.46);
insert into SHOT values (24, 14, 13, 21, 7.280, 73.13, 2.90, 7.292, 72.58, 2.67, 0.076, 353.64, -21.86);
insert into SHOT_FLAG values(24, 'dpl');
insert into SHOT values (25, 14, 15, 21, 10.140, 227.53, 11.90, 10.135, 229.74, 11.84, 0.383, 319.05, -1.63);
insert into SHOT values (26, 15, 14, 21, 10.120, 52.03, -11.80, 10.135, 49.74, -11.84, 0.396, 322.80, -1.52);
insert into SHOT_FLAG values(26, 'dpl');
insert into SHOT values (27, 15, 16, 21, 6.990, 46.43, -13.20, 6.990, 46.43, -13.23, 0.004, 241.21, -77.32);
insert into SHOT values (28, 15, 17, 21, 9.270, 152.63, -1.40, 9.278, 156.89, -1.54, 0.690, 244.18, -1.95);
insert into SHOT values (29, 17, 15, 21, 9.280, 341.23, 1.70, 9.278, 336.89, 1.54, 0.703, 248.92, -2.06);
insert into SHOT_FLAG values(29, 'dpl');
insert into SHOT values (30, 17, 18, 21, 11.660, 189.13, 11.90, 11.716, 192.55, 11.92, 0.684, 276.29, 1.31);
insert into SHOT values (31, 17, 18, 21, 11.730, 189.43, 12.10, 11.716, 192.55, 11.92, 0.624, 281.51, -3.57);
insert into SHOT values (32, 18, 17, 21, 11.720, 15.63, -11.80, 11.716, 12.55, -11.92, 0.618, 283.29, -2.16);
insert into SHOT values (33, 18, 17, 21, 11.730, 15.93, -11.80, 11.716, 12.55, -11.92, 0.678, 282.68, -1.80);
insert into SHOT_FLAG values(33, 'dpl');
insert into SHOT values (34, 18, 19, 21, 4.050, 138.23, -8.50, 4.053, 138.24, -8.51, 0.003, 144.57, -23.73);
insert into SHOT values (35, 18, 20, 21, 12.790, 246.63, 1.90, 12.814, 246.88, 1.57, 0.096, 311.51, -50.55);
insert into SHOT values (36, 20, 18, 21, 12.820, 68.33, -1.50, 12.814, 66.88, -1.57, 0.325, 336.43, -2.54);
insert into SHOT_FLAG values(36, 'dpl');
insert into SHOT values (37, 20, 18, 21, 12.830, 65.33, -0.80, 12.814, 66.88, -1.57, 0.386, 159.38, -26.24);
insert into SHOT_FLAG values(37, 'dpl');
insert into SHOT values (38, 20, 18, 21, 12.830, 67.63, -1.50, 12.814, 66.88, -1.57, 0.170, 331.61, -4.78);
insert into SHOT_FLAG values(38, 'dpl');
insert into SHOT values (39, 20, 21, 21, 9.710, 142.43, 1.80, 9.796, 147.09, 1.70, 0.797, 228.56, -1.08);
insert into SHOT values (40, 21, 20, 21, 9.870, 331.73, -1.50, 9.796, 327.09, -1.70, 0.800, 234.01, -2.27);
insert into SHOT_FLAG values(40, 'dpl');
insert into SHOT values (41, 21, 22, 21, 10.370, 202.23, 8.60, 10.356, 205.81, 8.50, 0.641, 295.04, -1.85);
insert into SHOT values (42, 21, 22, 21, 10.350, 202.23, 8.60, 10.356, 205.81, 8.50, 0.641, 293.27, -1.58);
insert into SHOT values (43, 22, 21, 21, 10.340, 29.43, -8.40, 10.356, 25.81, -8.50, 0.646, 298.77, -1.73);
insert into SHOT_FLAG values(43, 'dpl');
insert into SHOT values (44, 22, 23, 21, 12.390, 266.23, 2.20, 12.379, 265.97, 2.08, 0.063, 165.91, -23.88);
insert into SHOT values (45, 23, 22, 21, 12.370, 85.73, -2.00, 12.379, 85.97, -2.08, 0.055, 166.75, -19.51);
insert into SHOT_FLAG values(45, 'dpl');
insert into SHOT values (46, 23, 24, 21, 9.660, 120.83, 5.60, 9.650, 123.84, 5.41, 0.506, 213.07, -3.70);
insert into SHOT values (47, 24, 23, 21, 9.640, 306.83, -5.20, 9.650, 303.84, -5.41, 0.503, 216.15, -4.14);
insert into SHOT_FLAG values(47, 'dpl');
insert into SHOT values (48, 24, 25, 21, 10.270, 290.83, -5.90, 10.274, 290.86, -5.87, 0.009, 343.33, 37.04);
insert into SHOT values (49, 24, 26, 21, 10.560, 147.43, -1.10, 10.511, 151.90, -1.25, 0.823, 243.09, -1.90);
insert into SHOT values (50, 26, 24, 21, 10.450, 336.43, 1.40, 10.511, 331.90, 1.25, 0.831, 248.44, -1.75);
insert into SHOT_FLAG values(50, 'dpl');
insert into SHOT values (51, 26, 27, 21, 2.010, 117.33, -38.20, 2.010, 119.02, -38.81, 0.051, 223.71, -19.46);
insert into SHOT values (52, 27, 26, 21, 2.020, 300.93, 39.40, 2.010, 299.02, 38.81, 0.057, 216.28, -22.88);
insert into SHOT_FLAG values(52, 'dpl');
insert into SHOT values (53, 27, 28, 21, 1.160, 183.83, -20.50, 1.127, 187.67, -21.34, 0.081, 303.20, -2.67);
insert into SHOT values (54, 28, 27, 21, 1.130, 11.83, 21.90, 1.127, 7.67, 21.34, 0.077, 280.44, -8.56);
insert into SHOT_FLAG values(54, 'dpl');
insert into SHOT values (55, 28, 29, 21, 5.910, 135.63, 22.70, 5.909, 139.09, 22.70, 0.329, 227.48, -0.12);
insert into SHOT values (56, 29, 28, 21, 5.940, 322.43, -22.50, 5.909, 319.09, -22.70, 0.321, 224.27, -1.22);
insert into SHOT values (57, 29, 28, 21, 5.920, 322.53, -22.50, 5.909, 319.09, -22.70, 0.329, 227.70, -2.53);
insert into SHOT values (58, 29, 28, 21, 5.940, 322.33, -22.60, 5.909, 319.09, -22.70, 0.311, 224.74, 0.50);
insert into SHOT_FLAG values(58, 'dpl');
insert into SHOT values (59, 29, 30, 21, 0.830, 97.63, 69.60, 0.833, 95.91, 69.51, 0.009, 21.09, 12.84);
insert into SHOT values (60, 30, 29, 21, 0.830, 272.93, -69.80, 0.833, 275.91, -69.51, 0.016, 346.17, -3.80);
insert into SHOT_FLAG values(60, 'dpl');
insert into SHOT values (61, 30, 31, 21, 1.650, 193.53, -5.70, 1.653, 193.36, -5.90, 0.008, 132.00, -47.68);
insert into SHOT values (62, 30, 32, 21, 13.760, 87.33, -7.20, 13.752, 87.94, -7.35, 0.150, 182.63, -13.65);
insert into SHOT values (63, 32, 30, 21, 13.750, 268.53, 7.50, 13.752, 267.94, 7.35, 0.145, 180.86, -13.88);
insert into SHOT_FLAG values(63, 'dpl');
insert into SHOT values (64, 32, 33, 21, 3.390, 340.03, 2.10, 3.393, 340.18, 2.03, 0.010, 49.40, -24.77);
insert into SHOT values (65, 32, 34, 21, 9.270, 200.43, -4.90, 9.271, 203.35, -4.95, 0.470, 291.89, -1.00);
insert into SHOT values (66, 34, 32, 21, 9.280, 26.33, 5.00, 9.271, 23.35, 4.95, 0.482, 293.82, -1.05);
insert into SHOT_FLAG values(66, 'dpl');
insert into SHOT values (67, 34, 35, 21, 2.980, 102.83, 1.60, 2.969, 104.04, 1.35, 0.065, 202.70, -11.77);
insert into SHOT values (68, 35, 34, 21, 2.970, 285.23, -1.20, 2.969, 284.04, -1.35, 0.062, 193.97, -7.18);
insert into SHOT_FLAG values(68, 'dpl');
insert into SHOT values (69, 35, 36, 21, 3.500, 295.13, -2.60, 3.502, 295.03, -2.62, 0.007, 223.14, -10.42);
insert into STATION values (1, '0', 19, 881894.77, 2244204.83, 432.23);
insert into STATION values (2, 'c3', 19, 881899.86, 2244209.37, 432.50);
insert into STATION values (3, '1', 19, 881899.12, 2244198.77, 430.19);
insert into STATION values (4, '2', 19, 881897.14, 2244192.16, 428.83);
insert into STATION values (5, '3', 19, 881895.69, 2244189.73, 428.93);
insert into STATION values (6, '4', 19, 881902.42, 2244177.13, 428.38);
insert into STATION values (7, '5', 19, 881910.97, 2244169.42, 432.95);
insert into STATION values (8, 'c8', 19, 881904.92, 2244171.68, 430.91);
insert into STATION values (9, '6', 19, 881907.01, 2244168.68, 430.63);
insert into STATION values (10, '7', 19, 881910.92, 2244158.69, 432.84);
insert into STATION values (11, '8', 19, 881925.68, 2244156.96, 435.76);
insert into STATION values (12, '9', 19, 881919.06, 2244148.26, 433.57);
insert into STATION values (13, '10', 19, 881917.76, 2244143.93, 434.89);
insert into STATION values (14, '11', 19, 881910.81, 2244141.75, 434.55);
insert into STATION values (15, '12', 19, 881903.24, 2244135.34, 436.63);
insert into STATION values (16, 'c7', 19, 881908.17, 2244140.03, 435.03);
insert into STATION values (17, '13', 19, 881906.88, 2244126.81, 436.38);
insert into STATION values (18, '14', 19, 881904.39, 2244115.62, 438.80);
insert into STATION values (19, 'c6', 19, 881907.06, 2244112.63, 438.20);
insert into STATION values (20, '15', 19, 881892.61, 2244110.59, 439.15);
insert into STATION values (21, '16', 19, 881897.93, 2244102.37, 439.44);
insert into STATION values (22, '17', 19, 881893.47, 2244093.15, 440.97);
insert into STATION values (23, '18', 19, 881881.13, 2244092.28, 441.42);
insert into STATION values (24, '19', 19, 881889.11, 2244086.93, 442.33);
insert into STATION values (25, 'c5', 19, 881879.56, 2244090.57, 441.28);
insert into STATION values (26, '20', 19, 881894.06, 2244077.66, 442.10);
insert into STATION values (27, '21', 19, 881895.43, 2244076.90, 440.84);
insert into STATION values (28, '22', 19, 881895.29, 2244075.86, 440.43);
insert into STATION values (29, '23', 19, 881898.86, 2244071.74, 442.71);
insert into STATION values (30, '24', 19, 881899.15, 2244071.71, 443.49);
insert into STATION values (31, 'c4', 19, 881898.77, 2244070.11, 443.32);
insert into STATION values (32, '25', 19, 881912.78, 2244072.20, 441.73);
insert into STATION values (33, 'c2', 19, 881911.63, 2244075.39, 441.85);
insert into STATION values (34, '26', 19, 881909.12, 2244063.72, 440.93);
insert into STATION values (35, '27', 19, 881912.00, 2244063.00, 441.00);
insert into STATION_FLAG values(35, 'fix');
insert into STATION values (36, 'c1', 19, 881908.83, 2244064.48, 440.84);