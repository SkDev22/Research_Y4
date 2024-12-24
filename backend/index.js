import express from "express";
import cors from "cors";
import { connectDB } from "./config/db.js";

const app = express();
const port = process.env.PORT || 5000;

app.use(express());
app.use(cors());

// DB Connection
connectDB();

// Routes
app.get("/", (req, res) => {
  res.send("App is working");
});

app.listen(port, () => {
  console.log(`App running on port ${port}`);
});
