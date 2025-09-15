'use client';

import { useState, useEffect } from 'react';

// The backend API URL. In development, this points to our local Flask server.
const API_URL = 'http://localhost:8080/api/dashboard';

export default function Home() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        setError(null);
        const response = await fetch(API_URL);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        setData(result);
      } catch (e: any) {
        setError(`Failed to fetch data: ${e.message}. Make sure the Python backend is running.`);
        console.error(e);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []); // The empty dependency array means this effect runs once on mount.

  return (
    <main className="flex min-h-screen flex-col items-center p-8 bg-gray-900 text-white">
      <h1 className="text-4xl font-bold mb-8">Trading Dashboard</h1>

      {loading && <p className="text-xl text-blue-400">Loading dashboard data...</p>}
      
      {error && <p className="text-xl text-red-500 bg-red-100 p-4 rounded-lg text-gray-900">{error}</p>}

      {data && (
        <div className="w-full max-w-6xl bg-gray-800 p-4 rounded-lg shadow-lg">
          <h2 className="text-2xl mb-4">Raw Dashboard Data</h2>
          <pre className="text-sm bg-gray-900 p-4 rounded overflow-auto">
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </main>
  );
}
