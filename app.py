from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import mysql.connector

app = FastAPI()

# Fully enable CORS to allow browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db_connection():
    return mysql.connector.connect(
        host="127.0.0.1",
        user="root",
        password="YOUR_DATABASE_PASSWORD",
        database="Project_Database"
    )

@app.get("/api/dashboard-data")
def get_dashboard_data():
    try:
        conn = get_db_connection()
       # Adding buffered=True is crucial when handling large amounts of data to prevent "Internal Loss of Connection" errors or database hangs.
        cursor = conn.cursor(dictionary=True, buffered=True)

        # 'LIMIT 500' was removed to fetch all data.
        query = """
            SELECT 
                j.Job_Title AS `عنوان الوظيفة (Job_Title)`,
                j.Category AS `التصنيف (Category)`,
                l.Country_Name AS `الدولة (Country_Name)`,
                f.Hourly_Rate_Lower AS `الحد الأدنى (Lower)`,
                f.Hourly_Rate_Upper AS `الحد الأعلى (Upper)`,
                f.Hourly_Rate_Avg AS `المتوسط (Avg)`,
                IF(a.Is_High_Demand = 1, 'Yes', 'No') AS `طلب عالي`
            FROM Fact_Job_Posting f
            JOIN Dim_Job j ON f.Job_Key = j.Job_Key
            JOIN Dim_Location l ON f.Location_Key = l.Location_Key
            JOIN Dim_Posting_Attributes a ON f.Attributes_Key = a.Attributes_Key;
        """
        cursor.execute(query)
        rows = cursor.fetchall()

        cursor.execute("SELECT COUNT(DISTINCT Country_Name) as country_count FROM Dim_Location")
        geo_count = cursor.fetchone()["country_count"]

        cursor.close()
        conn.close()

        return {"rows": rows, "total_countries": geo_count}
    except Exception as e:
        print("Database Error:", str(e))
        return {"rows": [], "total_countries": 0, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)






