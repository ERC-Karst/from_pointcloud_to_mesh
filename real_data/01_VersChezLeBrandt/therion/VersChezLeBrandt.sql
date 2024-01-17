create table SURVEY (ID integer, PARENT_ID integer, NAME varchar(16), FULL_NAME varchar(16), TITLE varchar(18));
create table CENTRELINE (ID integer, SURVEY_ID integer, TITLE varchar(4), TOPO_DATE date, EXPLO_DATE date, LENGTH real, SURFACE_LENGTH real, DUPLICATE_LENGTH real);
create table PERSON (ID integer, NAME varchar(1), SURNAME varchar(1));
create table EXPLO (PERSON_ID integer, CENTRELINE_ID integer);
create table TOPO (PERSON_ID integer, CENTRELINE_ID integer);
create table STATION (ID integer, NAME varchar(4), SURVEY_ID integer, X real, Y real, Z real);
create table STATION_FLAG (STATION_ID integer, FLAG char(3));
create table SHOT (ID integer, FROM_ID integer, TO_ID integer, CENTRELINE_ID integer, LENGTH real, BEARING real, GRADIENT real, ADJ_LENGTH real, ADJ_BEARING real, ADJ_GRADIENT real, ERR_LENGTH real, ERR_BEARING real, ERR_GRADIENT real);
create table SHOT_FLAG (SHOT_ID integer, FLAG char(3));
create table MAPS (ID integer, SURVEY_ID integer, NAME varchar(16), TITLE varchar(18), PROJID integer, LENGTH real, DEPTH real);
create table SCRAPS (ID integer, SURVEY_ID integer, NAME varchar(16), PROJID integer, MAX_DISTORTION real, AVG_DISTORTION real);
create table MAPITEMS (ID integer, TYPE integer, ITEMID integer);
insert into SURVEY values (1, 0, '', '', NULL);
 insert into CENTRELINE values (2, 1, NULL, NULL, NULL, 0.00, 0.00, 0.00);
 insert into SURVEY values (20, 1, 'verschezlebrandt', 'verschezlebrandt', 'verschezlebrandt');
 insert into CENTRELINE values (21, 20, NULL, NULL, NULL, 0.00, 0.00, 0.00);
 insert into CENTRELINE values (22, 20, NULL, '2023-07-12', NULL, 493.28, 0.00, 0.00);
 insert into SHOT values (1, 1, 2, 22, 5.170, 240.33, 5.50, 5.170, 240.30, 5.55, 0.005, 150.39, 56.36);
insert into SHOT values (2, 1, 3, 22, 9.910, 298.83, 10.30, 9.890, 298.41, 10.25, 0.075, 194.24, -9.16);
insert into SHOT values (3, 3, 1, 22, 9.880, 117.93, -10.10, 9.890, 118.41, -10.25, 0.086, 204.62, -18.58);
insert into SHOT values (4, 3, 4, 22, 11.970, 323.13, 12.70, 11.986, 322.57, 12.58, 0.118, 243.50, -10.55);
insert into SHOT values (5, 4, 3, 22, 12.000, 142.03, -12.50, 11.986, 142.57, -12.58, 0.113, 241.03, -6.48);
insert into SHOT values (6, 4, 5, 22, 11.370, 333.53, 16.20, 11.365, 332.92, 16.14, 0.117, 242.43, -5.95);
insert into SHOT values (7, 5, 6, 22, 9.190, 151.63, -15.90, 9.185, 151.61, -15.92, 0.007, 2.29, -18.61);
insert into SHOT values (8, 5, 4, 22, 11.370, 152.33, -16.10, 11.365, 152.92, -16.14, 0.113, 246.26, -3.53);
insert into SHOT values (9, 5, 7, 22, 0.960, 107.13, 11.10, 0.962, 107.28, 11.40, 0.006, 185.69, 63.81);
insert into SHOT values (10, 5, 8, 22, 9.760, 295.03, 6.90, 9.759, 294.18, 6.77, 0.145, 205.10, -8.96);
insert into SHOT values (11, 8, 5, 22, 9.750, 113.33, -6.60, 9.759, 114.18, -6.77, 0.148, 201.72, -11.48);
insert into SHOT values (12, 8, 9, 22, 15.040, 6.83, 6.50, 15.035, 6.73, 6.34, 0.051, 275.74, -57.45);
insert into SHOT values (13, 9, 8, 22, 15.040, 186.63, -6.20, 15.035, 186.73, -6.34, 0.045, 296.89, -53.27);
insert into SHOT values (14, 9, 10, 22, 18.570, 303.83, 2.10, 18.570, 303.86, 2.10, 0.010, 33.57, -2.65);
insert into SHOT values (15, 9, 11, 22, 18.570, 302.53, 2.10, 18.583, 302.58, 1.94, 0.055, 349.41, -67.13);
insert into SHOT values (16, 11, 9, 22, 18.590, 122.63, -1.80, 18.583, 122.58, -1.94, 0.050, 5.23, -67.59);
insert into SHOT values (17, 11, 12, 22, 18.570, 121.53, -1.50, 18.572, 121.50, -1.48, 0.012, 42.06, 30.55);
insert into SHOT values (18, 11, 13, 22, 16.460, 311.13, 13.60, 16.462, 310.64, 13.42, 0.145, 226.88, -20.29);
insert into SHOT values (19, 13, 11, 22, 16.460, 130.13, -13.20, 16.462, 130.64, -13.42, 0.157, 225.31, -23.03);
insert into SHOT values (20, 13, 14, 22, 2.890, 96.43, -8.80, 2.892, 96.43, -8.75, 0.003, 95.04, 46.91);
insert into SHOT values (21, 13, 15, 22, 13.380, 355.53, 17.20, 13.388, 355.20, 17.12, 0.076, 275.62, -12.60);
insert into SHOT values (22, 15, 13, 22, 13.390, 174.83, -17.00, 13.388, 175.20, -17.12, 0.088, 271.93, -16.67);
insert into SHOT values (23, 15, 16, 22, 8.540, 279.33, -2.00, 8.539, 278.90, -2.15, 0.068, 187.26, -18.90);
insert into SHOT values (24, 16, 15, 22, 8.530, 98.53, 2.20, 8.539, 98.90, 2.15, 0.056, 179.42, -7.61);
insert into SHOT values (25, 16, 17, 22, 7.390, 7.83, 12.50, 7.392, 7.80, 12.50, 0.004, 310.29, 7.73);
insert into SHOT values (26, 16, 18, 22, 18.390, 331.03, 6.70, 18.390, 330.67, 6.49, 0.132, 244.67, -29.79);
insert into SHOT values (27, 18, 16, 22, 18.390, 150.33, -6.30, 18.390, 150.67, -6.49, 0.126, 244.13, -29.59);
insert into SHOT values (28, 18, 19, 22, 14.300, 324.33, 12.20, 14.306, 324.34, 12.23, 0.010, 3.23, 56.91);
insert into SHOT values (29, 19, 20, 22, 7.370, 323.73, 23.00, 7.369, 323.76, 23.01, 0.003, 83.34, 5.12);
insert into SHOT values (30, 19, 21, 22, 11.390, 334.03, 20.30, 11.339, 333.84, 20.23, 0.064, 192.62, -29.80);
insert into SHOT values (31, 21, 19, 22, 11.290, 153.63, -20.20, 11.339, 153.84, -20.23, 0.063, 195.92, -20.02);
insert into SHOT values (32, 21, 22, 22, 5.110, 4.13, 16.30, 5.109, 3.98, 16.37, 0.015, 261.94, 23.48);
insert into SHOT values (33, 22, 21, 22, 5.110, 183.83, -16.30, 5.109, 183.98, -16.37, 0.014, 286.40, -24.12);
insert into SHOT values (34, 22, 23, 22, 9.840, 342.13, -2.40, 9.815, 341.88, -2.45, 0.050, 221.80, -9.08);
insert into SHOT values (35, 23, 22, 22, 9.790, 161.63, 2.50, 9.815, 161.88, 2.45, 0.050, 220.69, -8.04);
insert into SHOT values (36, 23, 24, 22, 4.640, 73.83, 25.30, 4.640, 73.81, 25.26, 0.004, 30.32, -55.61);
insert into SHOT values (37, 23, 25, 22, 15.570, 282.73, 13.20, 15.557, 282.27, 13.04, 0.131, 191.27, -20.33);
insert into SHOT values (38, 25, 23, 22, 15.550, 101.83, -12.90, 15.557, 102.27, -13.04, 0.122, 192.81, -18.41);
insert into SHOT values (39, 25, 26, 22, 8.840, 340.33, 11.10, 8.835, 339.98, 10.96, 0.058, 249.80, -22.32);
insert into SHOT values (40, 26, 25, 22, 8.820, 159.53, -10.90, 8.835, 159.98, -10.96, 0.070, 238.55, -10.00);
insert into SHOT values (41, 26, 27, 22, 4.520, 252.93, 17.50, 4.530, 252.91, 17.47, 0.010, 243.27, 4.54);
insert into SHOT values (42, 26, 28, 22, 17.290, 299.13, 17.40, 17.303, 298.64, 17.28, 0.145, 218.28, -12.09);
insert into SHOT values (43, 28, 26, 22, 17.310, 118.23, -17.10, 17.303, 118.64, -17.28, 0.132, 219.23, -22.42);
insert into SHOT values (44, 28, 29, 22, 5.520, 321.63, 3.90, 5.521, 321.49, 3.84, 0.015, 235.91, -21.51);
insert into SHOT values (45, 29, 28, 22, 5.510, 141.33, -3.80, 5.521, 141.49, -3.84, 0.019, 196.81, -14.80);
insert into SHOT values (46, 29, 30, 22, 3.410, 245.73, 26.70, 3.402, 245.71, 26.72, 0.008, 72.92, -16.24);
insert into SHOT values (47, 29, 31, 22, 7.150, 248.13, 36.40, 7.142, 248.22, 36.52, 0.019, 38.11, 21.95);
insert into SHOT values (48, 31, 29, 22, 7.140, 68.33, -36.70, 7.142, 68.22, -36.52, 0.025, 31.67, 42.06);
insert into SHOT values (49, 31, 32, 22, 7.300, 342.53, 55.30, 7.307, 342.79, 55.33, 0.020, 71.31, 24.20);
insert into SHOT values (50, 32, 31, 22, 7.310, 163.03, -55.40, 7.307, 162.79, -55.33, 0.020, 89.41, 21.13);
insert into SHOT values (51, 32, 33, 22, 0.680, 216.93, 7.10, 0.677, 216.53, 6.79, 0.007, 96.24, -36.52);
insert into SHOT values (52, 32, 34, 22, 1.190, 189.03, -0.80, 1.194, 188.67, -0.96, 0.009, 125.77, -22.15);
insert into SHOT values (53, 32, 35, 22, 2.210, 172.03, 0.10, 2.212, 171.94, 0.00, 0.005, 110.78, -45.35);
insert into SHOT values (54, 32, 36, 22, 3.320, 157.03, -3.20, 3.321, 156.91, -3.28, 0.008, 72.73, -34.98);
insert into SHOT values (55, 32, 37, 22, 5.190, 145.23, -4.40, 5.189, 145.23, -4.42, 0.002, 304.50, -54.63);
insert into SHOT values (56, 32, 38, 22, 0.450, 137.93, -5.30, 0.448, 137.73, -5.13, 0.003, 354.98, 30.81);
insert into SHOT values (57, 32, 39, 22, 5.390, 131.63, -2.60, 5.396, 131.62, -2.55, 0.008, 119.46, 36.15);
insert into SHOT values (58, 32, 40, 22, 3.720, 122.53, -3.50, 3.722, 122.58, -3.54, 0.004, 188.98, -40.53);
insert into SHOT values (59, 32, 41, 22, 2.420, 114.13, -1.50, 2.422, 114.13, -1.42, 0.004, 115.06, 53.91);
insert into SHOT values (60, 32, 42, 22, 1.260, 97.13, 3.00, 1.262, 97.29, 3.18, 0.006, 159.17, 44.73);
insert into SHOT values (61, 32, 43, 22, 1.060, 71.23, 13.00, 1.062, 71.39, 13.07, 0.004, 138.12, 26.29);
insert into STATION values (1, '0', 20, 526560.64, 198828.80, 1110.58);
insert into STATION values (2, 'C1', 20, 526556.17, 198826.25, 1111.08);
insert into STATION values (3, '1', 20, 526552.08, 198833.43, 1112.34);
insert into STATION values (4, '2', 20, 526544.97, 198842.72, 1114.95);
insert into STATION values (5, '3', 20, 526540.00, 198852.44, 1118.11);
insert into STATION values (6, '.', 20, 526544.20, 198844.67, 1115.59);
insert into STATION values (7, 'C2', 20, 526540.90, 198852.16, 1118.30);
insert into STATION values (8, '4', 20, 526531.16, 198856.41, 1119.26);
insert into STATION values (9, '5', 20, 526532.91, 198871.25, 1120.92);
insert into STATION values (10, '.', 20, 526517.50, 198881.59, 1121.60);
insert into STATION values (11, '6', 20, 526517.26, 198881.25, 1121.55);
insert into STATION values (12, '.', 20, 526533.09, 198871.55, 1121.07);
insert into STATION values (13, '7', 20, 526505.11, 198891.68, 1125.37);
insert into STATION values (14, 'c3', 20, 526507.95, 198891.36, 1124.93);
insert into STATION values (15, '8', 20, 526504.04, 198904.43, 1129.31);
insert into STATION values (16, '9', 20, 526495.61, 198905.75, 1128.99);
insert into STATION values (17, 'c4', 20, 526496.59, 198912.90, 1130.59);
insert into STATION values (18, '10', 20, 526486.66, 198921.68, 1131.07);
insert into STATION values (19, '11', 20, 526478.51, 198933.04, 1134.10);
insert into STATION values (20, 'c5', 20, 526474.50, 198938.51, 1136.98);
insert into STATION values (21, '12', 20, 526473.82, 198942.59, 1138.02);
insert into STATION values (22, '13', 20, 526474.16, 198947.48, 1139.46);
insert into STATION values (23, '14', 20, 526471.11, 198956.80, 1139.04);
insert into STATION values (24, 'c6', 20, 526475.14, 198957.97, 1141.02);
insert into STATION values (25, '15', 20, 526456.30, 198960.02, 1142.55);
insert into STATION values (26, '16', 20, 526453.33, 198968.17, 1144.23);
insert into STATION values (27, 'c7', 20, 526449.20, 198966.90, 1145.59);
insert into STATION values (28, '17', 20, 526438.83, 198976.09, 1149.37);
insert into STATION values (29, '18', 20, 526435.40, 198980.40, 1149.74);
insert into STATION values (30, 'c8', 20, 526432.63, 198979.15, 1151.27);
insert into STATION values (31, '19', 20, 526430.07, 198978.27, 1153.99);
insert into STATION values (32, '20', 20, 526428.84, 198982.24, 1160.00);
insert into STATION_FLAG values(32, 'fix');
insert into STATION values (33, '.', 20, 526428.44, 198981.70, 1160.08);
insert into STATION values (34, '.', 20, 526428.66, 198981.06, 1159.98);
insert into STATION values (35, '.', 20, 526429.15, 198980.05, 1160.00);
insert into STATION values (36, '.', 20, 526430.14, 198979.19, 1159.81);
insert into STATION values (37, '.', 20, 526431.79, 198977.99, 1159.60);
insert into STATION values (38, '.', 20, 526429.14, 198981.91, 1159.96);
insert into STATION values (39, '.', 20, 526432.87, 198978.66, 1159.76);
insert into STATION values (40, '.', 20, 526431.97, 198980.24, 1159.77);
insert into STATION values (41, '.', 20, 526431.05, 198981.25, 1159.94);
insert into STATION values (42, '.', 20, 526430.09, 198982.08, 1160.07);
insert into STATION values (43, '.', 20, 526429.82, 198982.57, 1160.24);
