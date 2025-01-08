// eslint-disable-next-line react/prop-types
const Steps = ({ no, step }) => {
  return (
    <div>
      <h1>
        <span className="bg-newRed-200 rounded-full p-1 text-white font-bold">
          {no}
        </span>{" "}
        {step}
      </h1>
    </div>
  );
};

export default Steps;
