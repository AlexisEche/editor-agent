"use client";

import React, { useState } from "react";

export default function Home() {
  const [tasks, setTasks] = useState<string[]>([]);
  const [newTask, setNewTask] = useState("");

  const addTask = () => {
    if (newTask.trim()) {
      setTasks([...tasks, newTask]);
      setNewTask("");
    }
  };

  return (
    <div
      style={{
        fontFamily: "Courier New",
        backgroundColor: "#f0f8ff",
        padding: "20px",
        color: "#000000",
      }}
    >
      <h1>Lista de Tareas</h1>
      <input
        className="task-input"
        type="text"
        placeholder="Añadir tarea..."
        value={newTask}
        onChange={(e) => setNewTask(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && addTask()}
        style={{ width: "300px", padding: "5px", color: "#000000" }}
      />
      <ul>
        {tasks.map((task, index) => (
          <li
            key={index}
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              margin: "10px 0",
              color: "#000000",
            }}
          >
            {task}
            <button
              type="button"
              onClick={() =>
                setTasks(tasks.filter((_, i) => i !== index))
              }
              style={{
                padding: "5px 10px",
                backgroundColor: "#ff6347",
                color: "white",
                border: "none",
                cursor: "pointer",
              }}
            >
              Eliminar
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
