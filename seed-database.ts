import { v4 as uuidv4 } from "uuid";
import { Ollama } from "@langchain/ollama";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { MongoClient } from "mongodb";
import { MongoDBAtlasVectorSearch } from "@langchain/mongodb";
import { z } from "zod";
import "dotenv/config";
import { OllamaEmbeddings } from "@langchain/ollama";

const client = new MongoClient(process.env.MONGODB_ATLAS_URI || "");

// Ollama model setup
const llm = new Ollama({
  model: "codellama:7b",
  baseUrl: "http://127.0.0.1:11434",
  temperature: 0.9, // Increase to allow more randomness
  topK: 50, // Limit vocabulary selection to encourage diversity
  topP: 0.9, // Nucleus sampling for more variation
});

// Ollama embeddings setup
const embeddings = new OllamaEmbeddings({
  model: "codellama:7b",
  baseUrl: "http://127.0.0.1:11434",
});

// Define the schema for a single employee
const EmployeeSchema = z.object({
  employee_id: z.string(),
  first_name: z.string(),
  last_name: z.string(),
  date_of_birth: z.string(),
  address: z.object({
    street: z.string(),
    city: z.string(),
    state: z.string(),
    postal_code: z.string(),
    country: z.string(),
  }),
  contact_details: z.object({
    email: z.string().email(),
    phone_number: z.string(),
  }),
  job_details: z.object({
    job_title: z.string(),
    department: z.string(),
    hire_date: z.string(),
    employment_type: z.string(),
    salary: z.number(),
    currency: z.string(),
  }),
  work_location: z.object({
    nearest_office: z.string(),
    is_remote: z.boolean(),
  }),
  reporting_manager: z.string().nullable(),
  skills: z.array(z.string()),
  performance_reviews: z.array(
    z.object({
      review_date: z.string(),
      rating: z.number(),
      comments: z.string(),
    })
  ),
  benefits: z.object({
    health_insurance: z.string(),
    retirement_plan: z.string(),
    paid_time_off: z.number(),
  }),
  emergency_contact: z.object({
    name: z.string(),
    relationship: z.string(),
    phone_number: z.string(),
  }),
  notes: z.string(),
});

type Employee = z.infer<typeof EmployeeSchema>;

// Create a parser for a single employee
const singleEmployeeParser = StructuredOutputParser.fromZodSchema(EmployeeSchema);

async function generateSyntheticData(): Promise<Employee[]> {
  const prompt = `You are a highly creative assistant generating diverse employee records with efficiency. Each time you run, ensure the output is entirely unique, avoiding any repetition these fields: employee_id, first_name, last_name, date_of_birth, address, contact_details, job_details, work_location, reporting_manager, skills, performance_reviews, benefits, emergency_contact, notes. 

Requirements:
- Generate a **realistic but unique** employee record with distinct values.
- Use uncommon names, diverse job titles, and varied email formats.
- Ensure a wide range of skills and employment details.
- Randomize birthdates and locations across different runs.
- Maintain logical consistency.
- IMPORTANT: Return a SINGLE JSON OBJECT, not an array.

${singleEmployeeParser.getFormatInstructions()}`;
  
  console.log("Generating synthetic data...");

  try {
    // Generate multiple employees one at a time
    const employees: Employee[] = [];
    const response = await llm.invoke(prompt);

    try {
      // First try to parse as a single employee
      const employee = await singleEmployeeParser.parse(response);
      
      // Add unique ID
      const employeeWithId = {
        ...employee,
        employee_id: uuidv4(),
      };

      employees.push(employeeWithId);
      return employees;
    } catch (parseError) {
      console.error("Error parsing LLM response:", parseError);
      
      // Try to extract JSON from the response
      try {
        // Look for JSON in the response
        const jsonMatch = response.match(/\{[\s\S]*\}/);
        if (jsonMatch) {
          const extractedJson = jsonMatch[0];
          const parsedJson = JSON.parse(extractedJson);
          
          // Validate against our schema
          const validatedEmployee = EmployeeSchema.parse(parsedJson);
          
          // Add unique ID
          const employeeWithId = {
            ...validatedEmployee,
            employee_id: uuidv4(),
          };
          
          return [employeeWithId];
        }
        
        // If we found an array instead of an object
        const arrayMatch = response.match(/\[[\s\S]*\]/);
        if (arrayMatch) {
          const extractedArray = arrayMatch[0];
          const parsedArray = JSON.parse(extractedArray);
          
          // Take the first item if it's an array
          if (Array.isArray(parsedArray) && parsedArray.length > 0) {
            const firstEmployee = parsedArray[0];
            
            // Validate against our schema
            const validatedEmployee = EmployeeSchema.parse(firstEmployee);
            
            // Add unique ID
            const employeeWithId = {
              ...validatedEmployee,
              employee_id: uuidv4(),
            };
            
            return [employeeWithId];
          }
        }
      } catch (jsonError) {
        console.error("Failed to extract valid JSON:", jsonError);
      }
      
      // If all else fails, create a fallback employee
      console.log("Creating fallback employee record");
      const fallbackEmployee: Employee = {
        employee_id: uuidv4(),
        first_name: "John",
        last_name: "Doe",
        date_of_birth: "1990-01-01",
        address: {
          street: "123 Main St",
          city: "Anytown",
          state: "CA",
          postal_code: "12345",
          country: "USA",
        },
        contact_details: {
          email: "john.doe@example.com",
          phone_number: "+1-555-123-4567",
        },
        job_details: {
          job_title: "Software Engineer",
          department: "Engineering",
          hire_date: "2020-01-01",
          employment_type: "Full-time",
          salary: 80000,
          currency: "USD",
        },
        work_location: {
          nearest_office: "San Francisco",
          is_remote: false,
        },
        reporting_manager: "Jane Smith",
        skills: ["JavaScript", "TypeScript", "React"],
        performance_reviews: [
          {
            review_date: "2021-01-01",
            rating: 4,
            comments: "Good performance overall.",
          },
        ],
        benefits: {
          health_insurance: "Yes",
          retirement_plan: "401k",
          paid_time_off: 15,
        },
        emergency_contact: {
          name: "Jane Doe",
          relationship: "Spouse",
          phone_number: "+1-555-987-6543",
        },
        notes: "Reliable employee with good technical skills.",
      };
      
      return [fallbackEmployee];
    }
  } catch (error) {
    console.error("Error generating data with Ollama:", error);
    throw error;
  }
}

async function createEmployeeSummary(employee: Employee): Promise<string> {
  // Fixed template strings with proper backticks
  const jobDetails = `${employee.job_details.job_title} in ${employee.job_details.department}`;
  const skills = employee.skills.join(", ");
  const performanceReviews = employee.performance_reviews
    .map((review) => `Rated ${review.rating} on ${review.review_date}: ${review.comments}`)
    .join(" ");
  const basicInfo = `${employee.first_name} ${employee.last_name}, born on ${employee.date_of_birth}`;
  const workLocation = `Works at ${employee.work_location.nearest_office}, Remote: ${employee.work_location.is_remote ? 'Yes' : 'No'}`;
  const notes = employee.notes;

  return `${basicInfo}. Job: ${jobDetails}. Skills: ${skills}. Reviews: ${performanceReviews}. Location: ${workLocation}. Notes: ${notes}`;
}

async function seedDatabase(): Promise<void> {
  try {
    await client.connect();
    await client.db("admin").command({ ping: 1 });
    console.log("Pinged your deployment. You successfully connected to MongoDB!");

    const db = client.db("hr_database");
    const collection = db.collection("employees");

    // Generate data
    const syntheticData = await generateSyntheticData();
    console.log(`Generated ${syntheticData.length} employee records`);

    // Create summaries
    const recordsWithSummaries = await Promise.all(
      syntheticData.map(async (record) => ({
        pageContent: await createEmployeeSummary(record),
        metadata: { ...record },
      }))
    );

    // Insert in batches
    const batchSize = 10;
    for (let i = 0; i < recordsWithSummaries.length; i += batchSize) {
      const batch = recordsWithSummaries.slice(i, i + batchSize);
      await MongoDBAtlasVectorSearch.fromDocuments(
        batch,
        embeddings,
        {
          collection,
          indexName: "vector_index",
          textKey: "embedding_text",
          embeddingKey: "embedding",
        }
      );
      console.log(`Successfully processed & saved batch starting with record ID: ${batch[0].metadata.employee_id}`);
    }

    console.log("Database seeding completed");
  } catch (error) {
    console.error("Error seeding database:", error);
  } finally {
    await client.close();
  }
}

seedDatabase().catch(console.error);