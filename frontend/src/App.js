import React, { useState } from "react";

function App() {
  const [result, setResult] = useState(null);

  const handleAdd = async () => {
    const res = await fetch("http://127.0.0.1:5000/api/add", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ a: 5, b: 7 })
    });
    const data = await res.json();
    setResult(data.result);
  };

  return (
    <div style={{ padding: "20px" }}>
      <h1>ML Model Comparison Dashboard</h1>
      <button onClick={handleAdd}>Add Numbers</button>
      {result !== null && <p>Result: {result}</p>}
    </div>
  );
}

export default App;
