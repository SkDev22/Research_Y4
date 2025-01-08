import { Button } from "./ui/button";
import { SiLinksys } from "react-icons/si";

const Navbar = () => {
  return (
    <div className=" bg-newViolot-500 px-40 py-3 top-0 font-Poppins">
      <nav className="flex justify-between items-center">
        <div className="flex justify-center items-center gap-3 ">
          <SiLinksys className="text-3xl text-newRed-200 bg-transparent border p-1 border-newRed-200 rounded-sm" />
          <h1 className="text-white text-xl font-bold">LodgeLink</h1>
        </div>
        <div>
          <ul className="flex justify-center items-center gap-8 text-white">
            <li>Home</li>
            <li>Boarding Houses</li>
            <li>About Us</li>
            <li>Contact Us</li>
          </ul>
        </div>
        <div>
          <Button className="bg-transparent border-2 border-newRed-200 hover:bg-transparent hover:border-white px-6 rounded-2xl">
            Login
          </Button>
        </div>
      </nav>
    </div>
  );
};

export default Navbar;
