# Run python3 database.py  --server 183.177.126.159 --database LIVEAdvanceWarudGhantaGadi --ulbname UlhasNagarMahanagarPalika --hostname localhost --filename VehicleWiseDutyReport --ReportTitle "VehicleWiseDutyReport" --FromDate '2024-06-01' --ToDate '2024-07-05' --VehicleQR 26 --ZoneId 0 --PanelId 0

import numpy as np
import pandas as pd
import warnings
import argparse
import os
warnings.filterwarnings("ignore")
import pymssql
import plotly.graph_objs as go
import datetime
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-ip", "--server", required=True, help="Server IP address")
ap.add_argument("-db", "--database", required=True, help="Database name")
ap.add_argument("-ulbname", "--ulbname", required=True, help="name of the ULB")
ap.add_argument("-hostname", "--hostname", required=True, help="name of the ULB")
ap.add_argument("-filename", "--filename", required=True, help="name of the File")
ap.add_argument("-ReportTitle", "--ReportTitle", required=True, help="name of the ULB")
ap.add_argument("-FromDate", "--FromDate", required=True, help="Starting Date")
ap.add_argument("-ToDate", "--ToDate", required=True, help="Ending Date")
ap.add_argument("-VehicleQR", "--VehicleQR", required=True, help="Vehicle QR ID")
ap.add_argument("-ZoneId", "--ZoneId", required=True, help="ZoneId")
ap.add_argument("-PanelId", "--PanelId", required=True, help="PanelId")
args = vars(ap.parse_args())

# HostName
hostname = args["hostname"]
# Directory
directory = args["ulbname"]
# Filename
filename = args["filename"]
# Report Title
reporttitle = args["ReportTitle"]
# Starting Date
starting_date = args["FromDate"]
# ending_date
ending_date = args["ToDate"]
# Vehicle Qr ID
qr_id = args["VehicleQR"]

if hostname == "localhost":

    # Parent Directory path
    parent_dir = "D:/Ulhasnagar_ICTSBMCMS_28-08-2025/SwachhBharatAbhiyan.CMS/Images/AI"

else:

    # Parent Directory path
    parent_dir = "D:/Publish/Ulhasnagar_ICTSBM_CMS/Images/AI"

# Path
path = os.path.join(parent_dir, directory)

try:
    os.mkdir(path)
except OSError as error:
    print(error)

server = args["server"]
database = args["database"]

ZoneId = args["ZoneId"]
PanelId = args["PanelId"]

query1 = """ WITH base_attendance AS (
    SELECT cast(DA.daDate as date) AS Date,
           DA.userId,
           DA.daID
    FROM Daily_Attendance DA WITH (NOLOCK)
    WHERE DA.EmployeeType IS NULL
      AND cast(DA.daDate as date) BETWEEN @from AND @to
      AND DA.userId = @userid
),
user_name AS (
    SELECT userId, userName FROM UserMaster WITH (NOLOCK)
),
gc_union AS (
    SELECT gcDate, userId, houseId, gcType, 0 as isNotScan
    FROM GarbageCollectionDetails WITH (NOLOCK)
    WHERE cast(gcDate as date) BETWEEN @from AND @to

    UNION ALL

    SELECT gcDate, userId, houseId, gcType, 1 as isNotScan
    FROM GarbageCollection_NotScan WITH (NOLOCK)
    WHERE cast(gcDate as date) BETWEEN @from AND @to
),
filtered_gc AS (
    SELECT G.*, hm.ZoneId, wd.PanelId
    FROM gc_union G
    LEFT JOIN HouseMaster hm ON hm.houseId = G.houseId
    LEFT JOIN WardNumber wd ON hm.WardNo = wd.Id
    WHERE (@ZoneId = 0 OR @ZoneId IS NULL OR hm.ZoneId = @ZoneId)
      AND (@PanelId = 0 OR @PanelId IS NULL OR wd.PanelId = @PanelId)
)
SELECT A.Date,
       A.userId AS emp_id,
       U.userName AS EmployeeName,
       MIN(CASE WHEN gcType = 1 THEN CAST(gcDate AS TIME) END) AS FirstHouseScan,
       MAX(CASE WHEN gcType = 1 THEN CAST(gcDate AS TIME) END) AS LastHouseScan,
       SUM(CASE WHEN gcType = 1 THEN 1 ELSE 0 END) AS TotalHouseCount,
       MIN(CASE WHEN gcType = 3 THEN CAST(gcDate AS TIME) END) AS FirstDumpScan,
       SUM(CASE WHEN gcType = 3 THEN 1 ELSE 0 END) AS TotalDumpTrip
FROM base_attendance A
LEFT JOIN user_name U ON U.userId = A.userId
LEFT JOIN filtered_gc G ON G.userId = A.userId AND cast(G.gcDate as date) = A.Date
GROUP BY A.Date, A.userId, U.userName
ORDER BY A.Date ASC;
"""

def df_server(server, database):
    conn = pymssql.connect(server=server, user='user',
                           password='userpaass', database=database)
    df = pd.read_sql_query(query1,conn)
    return df
df_data = df_server(server=server, database=database)   


formatted_time = datetime.datetime.now().strftime("%d %b %Y %I:%M %p")



df2=df_data[['Date','DutyOffTime','DutyOnTime']]
print(df2[:50])