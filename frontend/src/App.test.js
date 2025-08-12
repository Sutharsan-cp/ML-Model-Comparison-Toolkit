import { render, screen, fireEvent } from "@testing-library/react";
import App from "./App";

test("renders app title", () => {
  render(<App />);
  expect(screen.getByText(/ML Model Comparison Dashboard/i)).toBeInTheDocument();
});
