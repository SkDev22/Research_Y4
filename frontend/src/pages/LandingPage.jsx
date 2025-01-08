import { Button } from "@/components/ui/button";
import { Link } from "react-router-dom";
import { MdArrowCircleRight } from "react-icons/md";
import Home01 from "../assets/Home01.jpg";
import { IoHome } from "react-icons/io5";
import { Skeleton } from "@/components/ui/skeleton";
// import { RiAlipayFill } from "react-icons/ri";
import Building2 from "../assets/Building2.jpg";
import Steps from "@/components/Steps";
// import { GiBrain } from "react-icons/gi";
import { LuBrainCircuit } from "react-icons/lu";
import { FaSearchengin } from "react-icons/fa6";
import { TbView360Number } from "react-icons/tb";
import { SiGoogleanalytics } from "react-icons/si";
import { IoMdPricetags } from "react-icons/io";

// eslint-disable-next-line no-unused-vars, react/prop-types
const LandingPage = ({ no, step }) => {
  return (
    <div className="">
      {/* Header */}
      {/* Decorations */}
      <div className="w-[300px] h-[300px] bg-newRed-200 ml-[970px] rounded-full blur-3xl mt-16 absolute opacity-50"></div>
      <div className="w-[300px] h-[300px] bg-newRed-200 ml-[1050px] rounded-full blur-3xl mt-52 absolute opacity-50"></div>
      <div className="w-[160px] h-[200px] bg-newViolot-500 shadow-2xl border-l border-b border-opacity-30 border-newRed-200 ml-[850px] rounded-xl mt-[295px] z-10 absolute flex flex-col justify-between items-center px-2">
        <div className="flex-1 w-full flex justify-between items-center">
          <IoHome className="text-3xl p-1 bg-white text-newViolot-500 rounded-md" />
          <div>
            <h1 className="text-sm text-white">Malani Boarding</h1>
            <p className="text-[10px] text-white">
              starting from{" "}
              <span className="text-newRed-200 font-semibold">100k Rs.</span>
            </p>
          </div>
        </div>
        <div className="flex-1 w-full mt-3 ">
          <div className="flex items-center space-x-4">
            <Skeleton className="h-3 w-3 rounded-full bg-white opacity-5" />
            <div className="space-y-2">
              <Skeleton className="h-1 w-[100px] bg-white opacity-5" />
              <Skeleton className="h-1 w-[100px] bg-white opacity-5" />
            </div>
          </div>
          <div className="flex items-center space-x-4 mt-2">
            <Skeleton className="h-3 w-3 rounded-full bg-white opacity-5" />
            <div className="space-y-2">
              <Skeleton className="h-1 w-[100px] bg-white opacity-5" />
              <Skeleton className="h-1 w-[100px] bg-white opacity-5" />
            </div>
          </div>
          <div className="flex items-center space-x-4 mt-2">
            <Skeleton className="h-3 w-3 rounded-full bg-white opacity-5" />
            <div className="space-y-2">
              <Skeleton className="h-1 w-[100px] bg-white opacity-5" />
              <Skeleton className="h-1 w-[100px] bg-white opacity-5" />
            </div>
          </div>
        </div>
        <div className="flex-1 w-full">
          <Button className="w-full h-7 mt-6 bg-newRed-200 text-newViolot-500 text-sm">
            Book Now
          </Button>
        </div>
      </div>
      {/* Header Details */}
      <div className="w-full flex justify-between items-center px-40 bg-newViolot-500 pb-14 ">
        <div className="flex-1 pr-8 py-10 h-[500px]">
          <p className="text-5xl text-white font-bold leading-tight">
            Easily Find Best <br /> Boarding House <br /> With LodgeLink
          </p>
          <p className="my-6 text-white text-justify">
            Lorem ipsum dolor, sit amet consectetur adipisicing elit. Unde,
            quos! Aut harum sed cumque ad. Animi est labore fugit ipsam, iste
            quam dolorem delectus nam dolores vero. Doloribus, aspernatur
            inventore? Lorem ipsum dolor, sit amet consectetur adipisicing elit.
            Unde, quos! Aut harum sed cumque ad. Animi est labore fugit ipsam,
            iste quam dolorem delectus nam dolores vero. Doloribus, aspernatur
            inventore?
          </p>
          <div className="flex justify-start gap-10 items-center">
            <Button className="bg-newRed-200 text-newViolot-500">
              Get Started
            </Button>
            <Link className="flex justify-center items-center gap-2 text-white">
              <MdArrowCircleRight className="text-lg" />
              Explore More
            </Link>
          </div>
        </div>
        <div className="flex-1">
          <img
            src={Home01}
            alt=""
            className="h-[450px] bg-cover w-[350px] rounded-tl-full rounded-tr-full relative ml-auto"
          />
        </div>
      </div>

      {/* Services */}
      <div className="px-40">
        <div className="w-full flex justify-between items-center h-[100px] -mt-9 bg-white shadow-lg rounded-xl mx-auto text-slate-500">
          <div className="flex-1 flex flex-col justify-center items-center hover:text-newRed-200 cursor-default">
            <LuBrainCircuit className="text-4xl" />
            <h1 className="font-semibold">AI Powered</h1>
          </div>
          <div className="flex-1 flex flex-col justify-center items-center hover:text-newRed-200 cursor-default">
            <FaSearchengin className="text-4xl" />
            <h1 className="font-semibold">Advanced Search</h1>
          </div>
          <div className="flex-1 flex flex-col justify-center items-center hover:text-newRed-200 cursor-default">
            <TbView360Number className="text-4xl" />
            <h1 className="font-semibold">360 View Experience</h1>
          </div>
          <div className="flex-1 flex flex-col justify-center items-center hover:text-newRed-200 cursor-default">
            <SiGoogleanalytics className="text-4xl" />
            <h1 className="font-semibold">Real-time Analytics</h1>
          </div>
          <div className="flex-1 flex flex-col justify-center items-center hover:text-newRed-200 cursor-default">
            <IoMdPricetags className="text-4xl" />
            <h1 className="font-semibold">Dynamic Pricing</h1>
          </div>
        </div>
      </div>
      {/* Steps */}
      <div className="px-40 my-20 min-h-screen">
        <div className="flex justify-between items-center">
          <div className="flex-1 p-3 pl-0">
            <img
              src={Building2}
              alt=""
              className="w-full h-[500px] rounded-lg"
            />
          </div>
          <div className="flex-1 flex flex-col justify-center gap-5 p-20 pr-0">
            <p className="text-3xl font-semibold">
              Website Platform that Provides <br /> Best Boarding Houses
            </p>
            <p className="text-justify">
              Lorem ipsum dolor sit amet consectetur adipisicing elit.
              Cupiditate omnis voluptatum dolore incidunt iste architecto at,
              asperiores nisi quo sapiente ea numquam, saepe commodi, impedit
              reprehenderit molestias officiis culpa magni.
            </p>
            <div className="grid grid-cols-2 gap-5">
              <Steps no="01" step="Register to the website" />
              <Steps no="02" step="Search Boarding Houses" />
              <Steps no="03" step="Explore Boarding Details" />
              <Steps no="04" step="Explore 360 view" />
              <Steps no="05" step="Chat with owner" />
              <Steps no="06" step="Book the boarding house" />
              <Steps no="07" step="Explore Analytics (For Owners)" />
            </div>
            <Button className="mr-auto w-32">Try Now</Button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LandingPage;
