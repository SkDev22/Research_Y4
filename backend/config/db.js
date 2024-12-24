import mongoose from "mongoose";

export const connectDB = async () => {
  await mongoose
    .connect(
      "mongodb+srv://kalharasahan78:TfQrulmyONCSSkvp@lodgelink.fnu2i.mongodb.net/?retryWrites=true&w=majority&appName=LodgeLink"
    )
    .then(() => {
      console.log("DB Connected");
    });
};
